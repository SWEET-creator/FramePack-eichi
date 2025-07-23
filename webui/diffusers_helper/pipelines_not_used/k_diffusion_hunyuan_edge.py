import torch
import math
import torch.nn.functional as F
import cv2
import numpy as np

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


def build_laplacian_pyramid(tensor, levels=3):
    """
    Laplacian Pyramidを構築
    """
    pyramid = []
    current = tensor
    
    for _ in range(levels):
        # ダウンサンプリング
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        # アップサンプリング
        up = F.interpolate(down, size=current.shape[-2:], mode='bilinear', align_corners=False)
        # Laplacianレイヤーを計算
        laplacian = current - up
        pyramid.append(laplacian)
        current = down
    
    pyramid.append(current)  # 最下層のガウシアンを追加
    return pyramid

def compute_edge_gradients_from_images(images, edge_strength=1.0):
    """
    画像空間でLaplacian Pyramidを使用してエッジ検出を行う
    """
    B, C, T, H, W = images.shape
    edge_gradients = torch.zeros_like(images)
    
    for b in range(B):
        for t in range(T):
            # RGB画像を取得 (3チャンネル)
            rgb_frame = images[b, :, t]  # (3, H, W)
            
            # グレースケールに変換
            gray_frame = 0.299 * rgb_frame[0] + 0.587 * rgb_frame[1] + 0.114 * rgb_frame[2]
            gray_frame = gray_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Laplacian Pyramidを構築
            pyramid = build_laplacian_pyramid(gray_frame, levels=3)
            
            # 各レベルのLaplacianを重み付きで合成
            edge_map = torch.zeros_like(gray_frame)
            for i, laplacian in enumerate(pyramid[:-1]):
                # 元のサイズにリサイズ
                if laplacian.shape != gray_frame.shape:
                    laplacian = F.interpolate(laplacian, size=gray_frame.shape[-2:], mode='bilinear', align_corners=False)
                
                # 重み付きで合成 (高周波成分を重視)
                weight = (i + 1) * 0.5
                edge_map += weight * torch.abs(laplacian)
            
            # エッジマップを正規化して3チャンネルに拡張
            edge_map = edge_map.squeeze()
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
            edge_map_3ch = edge_map.unsqueeze(0).expand(3, -1, -1)
            edge_gradients[b, :, t] = edge_map_3ch * edge_strength
    
    return edge_gradients


def apply_edge_enhancement_guidance(latents, vae, edge_strength=1.0, guidance_scale=1.0):
    """
    エッジ強調ガイダンスを適用（画像空間でエッジ検出を行う）
    """
    # データ型とデバイスを保存
    original_dtype = latents.dtype
    original_device = latents.device
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    
    # latentsを画像空間にデコード
    with torch.no_grad():
        # VAEのデバイスとdtypeに合わせる
        latents_for_vae = latents.to(device=vae_device, dtype=vae_dtype)
        
        # latentsを正規化範囲に調整
        latents_normalized = latents_for_vae / 0.18215
        images = vae.decode(latents_normalized).sample  # (-1, 1)の範囲
        
        # 画像を(0, 1)の範囲に正規化
        images = (images + 1.0) / 2.0
        images = images.to(dtype=torch.float32, device=original_device)  # エッジ検出用
    
    # 画像空間でエッジ勾配を計算
    edge_gradients_image = compute_edge_gradients_from_images(images, edge_strength)
    
    # エッジ勾配を適用（より強い効果のためスケールを調整）
    enhanced_images = images + guidance_scale * 0.3 * edge_gradients_image
    
    # 画像を(-1, 1)の範囲に戻す
    enhanced_images = torch.clamp(enhanced_images * 2.0 - 1.0, -1.0, 1.0)
    enhanced_images = enhanced_images.to(device=vae_device, dtype=vae_dtype)  # VAEの要件に戻す
    
    # 画像をlatent空間に再エンコード
    with torch.no_grad():
        enhanced_latents = vae.encode(enhanced_images).latent_dist.sample()
        enhanced_latents = enhanced_latents * 0.18215
        enhanced_latents = enhanced_latents.to(dtype=original_dtype, device=original_device)
    
    return enhanced_latents, edge_gradients_image

def apply_sequential_guidance(model, latents, sigmas, step_idx, edge_strength=1.0, guidance_scale=1.0):
    """
    各ステップで逐次的にガイダンスを適用（推論中は潜在空間でのみ軽微な処理）
    """
    # 推論中は計算負荷を軽減するため、単純な処理のみ行う
    # 実際のエッジ強調は最終段階で画像空間で行う
    return latents, torch.zeros_like(latents)

@torch.inference_mode()
def sample_hunyuan(
        transformer,
        vae=None,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        latents=None,
        denoise_strength=1.0,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        edge_enhancement_strength=0.0,
        edge_guidance_scale=1.0,
        **kwargs,
):
    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    if denoise_strength < 1.0 and latents is not None:
        noise = torch.randn_like(latents)

        # flux_muを使用してノイズレベルを計算
        seq_length = latents.shape[2] * latents.shape[3] * latents.shape[4] // 4
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
        sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
        
        # denoise_strengthに基づいて適切なノイズスケールを計算
        init_step = min(int(num_inference_steps * denoise_strength), num_inference_steps)
        start_step = max(num_inference_steps - init_step, 0)
        noise_scale = sigmas[start_step]  # 開始ステップのノイズスケールを使用
        
        print("[DEBUG] mu:", mu)
        print("[DEBUG] sigmas:", sigmas)
        print("[DEBUG] start_step:", start_step)
        print("[DEBUG] noise_scale:", noise_scale)
        
        original_latents = latents.clone()
        latents = latents * (1 - noise_scale) + noise_scale * noise
        print("latents shape:", latents.shape)
    else:
        latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)

    k_model = fm_wrapper(transformer)

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)

    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    # エッジ強調ガイダンスのコールバック関数を定義（逐次的に適用）
    def edge_guidance_callback(d):
        if edge_enhancement_strength > 0.0:
            x = d['denoised']
            step_idx = d['i']
            sigma = sigmas[step_idx] if step_idx < len(sigmas) else sigmas[-1]
            
            # 現在のステップでのノイズレベルに基づいてガイダンス強度を調整
            adaptive_strength = edge_enhancement_strength * (0.3 + 0.7 * (1.0 - sigma.item()))
            
            if adaptive_strength > 0.01:  # 最小閾値を設定
                try:
                    enhanced_x, _ = apply_edge_enhancement_guidance(
                        x, vae, adaptive_strength, edge_guidance_scale
                    )
                    d['denoised'] = enhanced_x
                    
                    if step_idx % 5 == 0:
                        change_magnitude = (enhanced_x - x).abs().mean().item()
                        print(f"[DEBUG] Step {step_idx}: Edge enhancement applied, change: {change_magnitude:.6f}, strength: {adaptive_strength:.3f}")
                        
                except Exception as e:
                    print(f"[WARNING] Edge enhancement failed at step {step_idx}: {e}")
                    # エラーが発生した場合は元のlatentsをそのまま使用
        
        return d
    
    # オリジナルのコールバックと組み合わせ
    original_callback = callback
    def combined_callback(d):
        d = edge_guidance_callback(d)
        if original_callback is not None:
            return original_callback(d)
        return d
    
    callback = combined_callback if edge_enhancement_strength > 0.0 else original_callback

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':
        # start_stepから始まるようにsigmasを調整
        if denoise_strength < 1.0:
            sigmas = sigmas[start_step:]
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    # エッジ強調は逐次的にコールバック内で適用済みのため、最終段階では追加処理なし
    if edge_enhancement_strength > 0.0:
        print(f"[DEBUG] Edge enhancement completed during sampling process")

    return results
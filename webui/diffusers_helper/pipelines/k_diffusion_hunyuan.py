import torch
import math
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import os
from PIL import Image

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size
from torch.utils.checkpoint import checkpoint

MAIN_GPU = "cuda:1"          # 拡散パイプライン用（30+ GiB）
GPU_VAE = "cuda:2"           # VAE + auto-refine 用（数GiB）

# GPU配置戦略:
# - MAIN_GPU: transformer, latents, 拡散アクティベーション（大容量メモリ使用）
# - GPU_VAE: VAE, auto-refineモデル（チャンクベース処理で40MiB確保）
# - CPU経由でのチャンクコピーにより親テンソル参照を切断してメモリ効率化  

# auto-refineモデルのインポートパス
sys.path.append('/root/auto-refine')
try:
    from degradation_model import create_model
except ImportError:
    print("Warning: auto-refine degradation_model not found")


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


def load_auto_refine_model(checkpoint_path="/root/auto-refine/outputs/outputs_spatial/best_model.pth", 
                           model_type='unet', device='cuda', mode="image"):
    """
    auto-refineモデルをロードする
    mode: "image" = ピクセル空間(3ch), "latent" = latent空間(16ch)
    """
    print(f"[DEBUG] load_auto_refine_model called with:")
    print(f"[DEBUG]   checkpoint_path: {checkpoint_path}")
    print(f"[DEBUG]   model_type: {model_type}")
    print(f"[DEBUG]   device: {device}")
    print(f"[DEBUG]   mode: {mode}")
    
    print(f"[DEBUG] Checking if 'create_model' exists in globals...")
    if 'create_model' not in globals():
        print("[DEBUG] ERROR: 'create_model' not found in globals()")
        print("Warning: auto-refine model not available, skipping heatmap guidance")
        return None
    
    print("[DEBUG] 'create_model' found in globals, proceeding...")
    
    try:
        print(f"[DEBUG] Creating device object...")
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] Device object created: {device_obj}")
        
        # モードに応じてチャンネル数を設定
        if mode == "image":
            n_channels = 3  # RGBピクセル空間
            print(f"[DEBUG] Using image mode: n_channels = {n_channels}")
        elif mode == "latent":
            n_channels = 16  # HunyuanVideo latent空間
            print(f"[DEBUG] Using latent mode: n_channels = {n_channels}")
        else:
            print(f"[DEBUG] ERROR: Unsupported mode: {mode}")
            raise ValueError(f"Unsupported mode: {mode}. Use 'image' or 'latent'.")
        
        # モデルを作成
        print(f"[DEBUG] Creating model with create_model({model_type}, n_channels={n_channels}, n_classes=1)...")
        model = create_model(model_type, n_channels=n_channels, n_classes=1)
        print(f"[DEBUG] Model created successfully: {type(model)}")
        
        # チェックポイントをロード
        print(f"[DEBUG] Checking if checkpoint exists at: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            print(f"[DEBUG] Checkpoint found, loading...")
            checkpoint = torch.load(checkpoint_path, map_location=device_obj)
            print(f"[DEBUG] Checkpoint loaded, keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                print(f"[DEBUG] Loading model state dict...")
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[DEBUG] Model state dict loaded successfully")
            else:
                print(f"[DEBUG] ERROR: 'model_state_dict' key not found in checkpoint")
                print(f"[DEBUG] Available keys: {list(checkpoint.keys())}")
                return None
            
            print(f"[DEBUG] Moving model to device: {device_obj}")
            model.to(device_obj)
            model.eval()
            print(f"[DEBUG] Model successfully loaded and moved to device")
            print(f"Auto-refine model ({mode} mode) loaded from {checkpoint_path}")
            return model, device_obj
        else:
            print(f"[DEBUG] ERROR: Checkpoint not found at {checkpoint_path}")
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return None
    except Exception as e:
        print(f"[DEBUG] EXCEPTION in load_auto_refine_model: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        print(f"Warning: Failed to load auto-refine model: {e}")
        return None

def generate_heatmap_guidance(images, auto_refine_model, device):
    """
    auto-refineモデルを使用してヒートマップガイダンスを生成
    """
    if auto_refine_model is None:
        return torch.zeros_like(images)
    
    B, C, T, H, W = images.shape
    heatmap_guidance = torch.zeros_like(images)
    
    with torch.no_grad():
        for b in range(B):
            for t in range(T):
                # フレームを取得 (3, H, W)
                frame = images[b, :, t]
                
                # フレームを(0, 1)範囲にクランプ
                frame = torch.clamp(frame, 0, 1)
                
                # バッチ次元を追加 (1, 3, H, W)
                frame_batch = frame.unsqueeze(0).to(device)
                
                try:
                    # auto-refineモデルで推論
                    heatmap_pred = auto_refine_model(frame_batch)  # (1, 1, H, W)
                    
                    # ヒートマップを3チャンネルに拡張
                    heatmap_3ch = heatmap_pred.squeeze(0).expand(3, -1, -1)  # (3, H, W)
                    
                    # 元のデバイスに戻す
                    heatmap_guidance[b, :, t] = heatmap_3ch.to(images.device)
                    
                except Exception as e:
                    print(f"Warning: Heatmap generation failed for batch {b}, frame {t}: {e}")
                    # エラー時はゼロマップを使用
                    heatmap_guidance[b, :, t] = torch.zeros_like(frame)
    
    return heatmap_guidance


# @torch.inference_mode()
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
        device=None,  # 拡散処理用デバイス。Noneの場合はMAIN_GPU (cuda:0)を使用
        negative_kwargs=None,
        callback=None,
        heatmap_guidance_strength=0.0,
        heatmap_guidance_scale=1.0,
        heatmap_guidance_steps=5,  # ★ 最後のnステップでのみガイダンスを適用
        auto_refine_checkpoint="/root/auto-refine/outputs/outputs_spatial/best_model.pth",
        degradation_mode="image",  # ★ 新しいパラメータ: "image" または "latent"
        **kwargs,
):
    # デフォルトデバイスをMAIN_GPUに設定（拡散処理用）
    if device is None:
        device = MAIN_GPU
        print(f"[DEBUG] Setting default device to {MAIN_GPU} for diffusion pipeline")
    
    print(f"[DEBUG] sample_hunyuan called with device: {device}, dtype: {dtype}")
    print(f"[DEBUG] GPU allocation: MAIN_GPU={MAIN_GPU} (diffusion), GPU_VAE={GPU_VAE} (VAE/auto-refine)")
    print(f"[DEBUG] generator device: {generator.device if generator is not None else 'None'}")
    print(f"[DEBUG] latents provided: {latents is not None}")
    if latents is not None:
        print(f"[DEBUG] provided latents device: {latents.device}")

    # auto-refineモデルをロード（ヒートマップガイダンスが有効な場合）
    auto_refine_model = None
    auto_refine_device = None
    if heatmap_guidance_strength > 0.0:
        # auto-refineモデルはGPU_VAEに配置
        model_result = load_auto_refine_model(auto_refine_checkpoint, 'unet', GPU_VAE, degradation_mode)
        if model_result is not None:
            auto_refine_model, auto_refine_device = model_result
            print(f"[DEBUG] Using degradation model in {degradation_mode} mode on {auto_refine_device}")
            print(f"[DEBUG] Heatmap guidance will be applied in last {heatmap_guidance_steps} steps out of {num_inference_steps} total steps")
            print(f"[DEBUG] Guidance will start from step {num_inference_steps - heatmap_guidance_steps}")

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    if denoise_strength < 1.0 and latents is not None:
        # 提供されたlatentsを確実に正しいデバイスに配置
        if latents.device != device:
            print(f"[DEBUG] Moving provided latents from {latents.device} to {device}")
            latents = latents.to(device)
        
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
        # 拡散用のlatentsはMAIN_GPUで初期化
        print(f"[DEBUG] Initializing latents on {device}")
        
        # generatorのデバイス処理
        if generator is not None:
            if generator.device.type == 'cpu' and device != 'cpu':
                print(f"[DEBUG] Moving generator from CPU to {device}")
                # PyTorchのGeneratorは直接移動できないので、新しいgeneratorを作成
                try:
                    seed = generator.initial_seed()
                    new_generator = torch.Generator(device=device)
                    new_generator.manual_seed(seed)
                    generator = new_generator
                    print(f"[DEBUG] Generator moved successfully with seed {seed}")
                except Exception as e:
                    print(f"[DEBUG] Generator move failed: {e}, using None generator")
                    generator = None
            elif generator.device != torch.device(device):
                print(f"[DEBUG] Generator device mismatch: {generator.device} vs {device}")
                try:
                    seed = generator.initial_seed()
                    new_generator = torch.Generator(device=device)
                    new_generator.manual_seed(seed)
                    generator = new_generator
                    print(f"[DEBUG] Generator recreated on {device} with seed {seed}")
                except Exception as e:
                    print(f"[DEBUG] Generator recreation failed: {e}, using None generator")
                    generator = None
        
        latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), 
                             generator=generator, device=device, dtype=dtype)

    print(f"[DEBUG] latents dtype: {latents.dtype}, shape: {latents.shape}")
    
    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    print(f"[DEBUG] mu: {mu}, seq_length: {seq_length}")
    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device=device, dtype=torch.float32)
    print(f"[DEBUG] sigmas dtype: {sigmas.dtype}, shape: {sigmas.shape}")

    print(f"[DEBUG] Moving transformer to {device}")
    transformer = transformer.to(device)
    print(f"[DEBUG] Creating k_model wrapper")
    k_model = fm_wrapper(transformer)
    print(f"[DEBUG] k_model created successfully on {device}")

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=dtype)
        initial_latent = initial_latent.to(device=device, dtype=dtype)
        latents = initial_latent * (1.0 - first_sigma) + latents * first_sigma
        print(f"[DEBUG] Applied initial_latent, new latents dtype: {latents.dtype}")

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)
        print(f"[DEBUG] concat_latent dtype: {concat_latent.dtype}")

    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)
    print(f"[DEBUG] distilled_guidance dtype: {distilled_guidance.dtype}")
    
    # エッジ強化とヒートマップガイダンス用のcond_fn関数を定義
    def guidance_cond_fn(x_t, sigma, i):
        if heatmap_guidance_strength <= 0.0:
            return torch.zeros_like(x_t)
        
        # 最後のheatmap_guidance_stepsでのみガイダンスを適用
        if i < (num_inference_steps - heatmap_guidance_steps):
            if i % 5 == 0:  # 5ステップごとにスキップメッセージを表示
                print(f"[DEBUG] Step {i}/{num_inference_steps}: Skipping heatmap guidance (will start at step {num_inference_steps - heatmap_guidance_steps})")
            return torch.zeros_like(x_t)
        
        print(f"[DEBUG] Step {i}/{num_inference_steps}: Applying heatmap guidance ({degradation_mode} mode)")
        
        print(f"[DEBUG] Step {i}: x_t dtype: {x_t.dtype}, sigma dtype: {sigma.dtype}")
        
        with torch.inference_mode(False):
            # 勾配計算を有効化
            x_t = x_t.detach().clone().requires_grad_(True)
            
            with torch.enable_grad():
                try:
                    with torch.no_grad():
                        # 1. ε予測とx0_hat計算
                        eps = k_model(x_t, sigma, **sampler_kwargs).detach()
                    torch.set_grad_enabled(True)
                    print(f"[DEBUG] Step {i}: eps dtype: {eps.dtype}")
                    
                    # epsのNaN/Infチェック
                    if torch.isnan(eps).any():
                        print(f"[ERROR] Step {i}: NaN detected in eps prediction!")
                        print(f"[DEBUG] eps stats: min={eps.min().item():.6f}, max={eps.max().item():.6f}")
                        print(f"[DEBUG] x_t stats: min={x_t.min().item():.6f}, max={x_t.max().item():.6f}")
                        print(f"[DEBUG] sigma: {sigma}")
                        raise RuntimeError(f"NaN in eps prediction at step {i}")
                    if torch.isinf(eps).any():
                        print(f"[ERROR] Step {i}: Inf detected in eps prediction!")
                        print(f"[DEBUG] eps stats: min={eps.min().item():.6f}, max={eps.max().item():.6f}")
                        # Infを安全な値でクランプ
                        eps = torch.clamp(eps, min=-100.0, max=100.0)
                        print(f"[WARNING] Step {i}: Clamped Inf values in eps")
                    
                    # alpha計算の安全性強化
                    sigma_safe = torch.clamp(sigma, min=1e-8, max=1000.0)  # sigma=0を避ける
                    alpha = 1 / (sigma_safe**2 + 1)
                    
                    # alphaのNaN/Infチェック
                    if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                        print(f"[ERROR] Step {i}: Invalid alpha computation!")
                        print(f"[DEBUG] sigma: {sigma}, sigma_safe: {sigma_safe}")
                        print(f"[DEBUG] alpha: {alpha}")
                        raise RuntimeError(f"Invalid alpha at step {i}")
                    
                    alpha_sqrt = torch.sqrt(alpha.float()).to(alpha.dtype)
                    
                    # alpha_sqrtのNaN/Infチェック
                    if torch.isnan(alpha_sqrt).any() or torch.isinf(alpha_sqrt).any():
                        print(f"[ERROR] Step {i}: Invalid alpha_sqrt computation!")
                        print(f"[DEBUG] alpha: {alpha}, alpha_sqrt: {alpha_sqrt}")
                        raise RuntimeError(f"Invalid alpha_sqrt at step {i}")
                    
                    # x0_hat計算（安全性強化）
                    sigma_eps = sigma_safe * eps
                    x_t_corrected = x_t - sigma_eps
                    x0_hat = x_t_corrected * alpha_sqrt
                    
                    # x0_hatのNaN/Infチェック
                    if torch.isnan(x0_hat).any():
                        print(f"[ERROR] Step {i}: NaN detected in x0_hat computation!")
                        print(f"[DEBUG] x_t stats: min={x_t.min().item():.6f}, max={x_t.max().item():.6f}")
                        print(f"[DEBUG] eps stats: min={eps.min().item():.6f}, max={eps.max().item():.6f}")
                        print(f"[DEBUG] sigma: {sigma}, alpha: {alpha.item():.6f}, alpha_sqrt: {alpha_sqrt.item():.6f}")
                        print(f"[DEBUG] sigma_eps stats: min={sigma_eps.min().item():.6f}, max={sigma_eps.max().item():.6f}")
                        print(f"[DEBUG] x_t_corrected stats: min={x_t_corrected.min().item():.6f}, max={x_t_corrected.max().item():.6f}")
                        raise RuntimeError(f"NaN in x0_hat computation at step {i}")
                    if torch.isinf(x0_hat).any():
                        print(f"[ERROR] Step {i}: Inf detected in x0_hat computation!")
                        print(f"[DEBUG] x0_hat stats: min={x0_hat.min().item():.6f}, max={x0_hat.max().item():.6f}")
                        # Infを安全な値でクランプ
                        x0_hat = torch.clamp(x0_hat, min=-100.0, max=100.0)
                        print(f"[WARNING] Step {i}: Clamped Inf values in x0_hat")
                    
                    print(f"[DEBUG] Step {i}: x0_hat dtype: {x0_hat.dtype}")
                    print(f"[DEBUG] Step {i}: x0_hat stats: min={x0_hat.min().item():.6f}, max={x0_hat.max().item():.6f}")
                except Exception as e:
                    print(f"[DEBUG] Step {i}: Error in eps computation: {e}")
                    raise
                
                print("x_t.requires_grad:", x_t.requires_grad)
                print("eps.requires_grad:", eps.requires_grad)
                print("x0_hat.requires_grad:", x0_hat.requires_grad)
                # 2. degradation_modeに応じて画像を準備（チャンクベース処理）
                print(f"[DEBUG] Step {i}: Starting degradation mode processing: {degradation_mode}")
                img = None
                grad_buf = torch.zeros_like(x_t)  # 勾配蓄積用バッファ
                
                if degradation_mode == "image" and vae is not None:
                    print(f"[DEBUG] Step {i}: Entering image mode VAE processing")
                    try:
                        print(f"[DEBUG] Step {i}: Converting x0_hat to fp16 and cloning to avoid inference tensor issues")
                        x0_hat_fp16 = x0_hat.half().clone()  # clone()を追加してinference tensorを回避
                        print(f"[DEBUG] Step {i}: x0_hat_fp16.shape = {x0_hat_fp16.shape}, dim = {x0_hat_fp16.dim()}")
                        print(f"[DEBUG] Step {i}: x0_hat_fp16.requires_grad = {x0_hat_fp16.requires_grad}")
                        
                        # VAEの準備（勾配無効化）
                        print(f"[DEBUG] Step {i}: Preparing VAE on {GPU_VAE}")
                        vae_fp16 = vae.half().to(GPU_VAE).eval()
                        for p in vae_fp16.parameters():
                            p.requires_grad_(False)   # VAE重みの勾配は不要
                        print(f"[DEBUG] Step {i}: VAE prepared successfully")

                        if x0_hat_fp16.dim() == 5:                  # (B,16,T,H8,W8)
                            B, C16, T, H8, W8 = x0_hat_fp16.shape
                            chunk_T = min(1, T)  # チャンクサイズ（メモリに応じて調整）
                            print(f"[DEBUG] Step {i}: Processing {T} frames in chunks of {chunk_T}")
                            
                            for start in range(0, T, chunk_T):
                                end = min(start + chunk_T, T)
                                print(f"[DEBUG] Step {i}: Processing chunk {start}:{end}")
                                
                                # ---- 1) latent チャンク抽出 ----
                                print(f"[DEBUG] Step {i}: Extracting chunk to CPU to break parent tensor reference")
                                
                                # x0_hat_fp16のNaN/Infチェック（チャンク抽出前）
                                chunk_source = x0_hat_fp16[:, :, start:end, :, :]
                                if torch.isnan(chunk_source).any():
                                    print(f"[ERROR] Step {i}: NaN detected in chunk source before extraction!")
                                    print(f"[DEBUG] chunk_source stats: min={chunk_source.min().item():.6f}, max={chunk_source.max().item():.6f}")
                                    print(f"[DEBUG] x0_hat_fp16 full stats: min={x0_hat_fp16.min().item():.6f}, max={x0_hat_fp16.max().item():.6f}")
                                    # NaN発生時はスキップまたは安全な値で代替
                                    print(f"[WARNING] Step {i}: Skipping chunk {start}:{end} due to NaN in source")
                                    continue  # このチャンクをスキップ
                                
                                if torch.isinf(chunk_source).any():
                                    print(f"[ERROR] Step {i}: Inf detected in chunk source before extraction!")
                                    print(f"[DEBUG] chunk_source stats: min={chunk_source.min().item():.6f}, max={chunk_source.max().item():.6f}")
                                    # Infは安全な値でクランプ
                                    chunk_source = torch.clamp(chunk_source, min=-50.0, max=50.0)
                                    print(f"[WARNING] Step {i}: Clamped Inf values in chunk source")
                                
                                latent_chunk = chunk_source.clone().cpu()  # clone()でinference tensor回避
                                print(f"[DEBUG] Step {i}: Moving chunk to {GPU_VAE} as independent copy")
                                latent_chunk = latent_chunk.to(GPU_VAE, non_blocking=True)     # 独立コピー
                                
                                # GPU移動後の最終チェック
                                if torch.isnan(latent_chunk).any():
                                    print(f"[ERROR] Step {i}: NaN detected in latent_chunk after GPU move!")
                                    print(f"[WARNING] Step {i}: Skipping chunk {start}:{end} due to NaN after GPU move")
                                    continue
                                
                                latent_chunk.requires_grad_(True)  # gradient path を作る
                                print(f"[DEBUG] Step {i}: Chunk memory footprint: {latent_chunk.numel() * latent_chunk.element_size() / 1024**2:.1f} MB")
                                print(f"[DEBUG] Step {i}: latent_chunk.requires_grad = {latent_chunk.requires_grad}")
                                print(f"[DEBUG] Step {i}: latent_chunk range: [{latent_chunk.min().item():.6f}, {latent_chunk.max().item():.6f}]")
                                
                                # ---- 2) forward (decode + heat-map) ----
                                print(f"[DEBUG] Step {i}: Starting forward pass - VAE decode")
                                with torch.set_grad_enabled(True):
                                    # latent_chunkのNaN/Infチェック（VAE入力前）
                                    if torch.isnan(latent_chunk).any():
                                        print(f"[ERROR] Step {i}: NaN detected in latent_chunk before VAE decode!")
                                        print(f"[DEBUG] latent_chunk stats: min={latent_chunk.min().item():.6f}, max={latent_chunk.max().item():.6f}")
                                        raise RuntimeError(f"NaN in latent_chunk before VAE decode at step {i}")
                                    if torch.isinf(latent_chunk).any():
                                        print(f"[ERROR] Step {i}: Inf detected in latent_chunk before VAE decode!")
                                        raise RuntimeError(f"Inf in latent_chunk before VAE decode at step {i}")
                                    
                                    # VAEデコード前の正規化値チェックと安全な処理
                                    vae_scale = 0.18215
                                    normalized_latent = latent_chunk / vae_scale
                                    
                                    # 正規化後の極値をクランプして安全性を確保
                                    normalized_latent = torch.clamp(normalized_latent, min=-50.0, max=50.0)
                                    
                                    if torch.isnan(normalized_latent).any():
                                        print(f"[ERROR] Step {i}: NaN detected after normalization (divide by {vae_scale})!")
                                        raise RuntimeError(f"NaN after normalization at step {i}")
                                    if torch.isinf(normalized_latent).any():
                                        print(f"[ERROR] Step {i}: Inf detected after normalization!")
                                        raise RuntimeError(f"Inf after normalization at step {i}")
                                    
                                    try:
                                        img_chunk = vae_fp16.decode(normalized_latent).sample  # (B,3,Δt,H,W)
                                    except Exception as vae_error:
                                        print(f"[ERROR] Step {i}: VAE decode failed: {vae_error}")
                                        print(f"[DEBUG] normalized_latent stats: min={normalized_latent.min().item():.6f}, max={normalized_latent.max().item():.6f}")
                                        raise RuntimeError(f"VAE decode failed at step {i}: {vae_error}")
                                    
                                    # VAEデコード出力のNaN/Infチェック
                                    if torch.isnan(img_chunk).any():
                                        print(f"[ERROR] Step {i}: NaN detected in VAE decode output!")
                                        print(f"[DEBUG] img_chunk stats: min={img_chunk.min().item():.6f}, max={img_chunk.max().item():.6f}")
                                        raise RuntimeError(f"NaN in VAE decode output at step {i}")
                                    if torch.isinf(img_chunk).any():
                                        print(f"[ERROR] Step {i}: Inf detected in VAE decode output!")
                                        raise RuntimeError(f"Inf in VAE decode output at step {i}")
                                    
                                    img_chunk = ((img_chunk + 1) / 2).to(x_t.dtype)             # 0-1 域へ
                                    
                                    # 0-1正規化後のNaN/Infチェック
                                    if torch.isnan(img_chunk).any():
                                        print(f"[ERROR] Step {i}: NaN detected after 0-1 normalization!")
                                        print(f"[DEBUG] img_chunk stats: min={img_chunk.min().item():.6f}, max={img_chunk.max().item():.6f}")
                                        raise RuntimeError(f"NaN after 0-1 normalization at step {i}")
                                    if torch.isinf(img_chunk).any():
                                        print(f"[ERROR] Step {i}: Inf detected after 0-1 normalization!")
                                        raise RuntimeError(f"Inf after 0-1 normalization at step {i}")
                                    
                                    img_chunk = img_chunk.clone() # 画像のコピー（inference tensor回避）
                                    print(f"[DEBUG] Step {i}: VAE decode completed, img_chunk.shape = {img_chunk.shape}")
                                    print(f"[DEBUG] Step {i}: img_chunk.requires_grad = {img_chunk.requires_grad}")
                                    print(f"[DEBUG] Step {i}: img_chunk range: [{img_chunk.min().item():.6f}, {img_chunk.max().item():.6f}]")
                                    
                                    # フレームごと heat-map → loss
                                    loss = torch.zeros(1, device=latent_chunk.device, dtype=x_t.dtype, requires_grad=True)
                                    chunk_frames = img_chunk.size(2)
                                    print(f"[DEBUG] Step {i}: Processing {chunk_frames} frames in chunk")
                                    
                                    for t_idx in range(chunk_frames):
                                        print(f"[DEBUG] Step {i}: Processing frame {t_idx}/{chunk_frames}")
                                        frame = (
                                            img_chunk[:, :, t_idx]          # (B,3,H,W)  still inference
                                                .clone()                    # new storage – but still inference
                                                .requires_grad_(True)       # and re-attach to graph we want
                                        )
                                        
                                        print(f"[DEBUG] Step {i}: Running auto_refine_model on frame")
                                        
                                        try:
                                            # フレームを安全な範囲にクランプ
                                            safe_frame = torch.clamp(frame, min=0.0, max=1.0)
                                            heatmap = auto_refine_model(safe_frame)      # (B,1,H,W)
                                        except Exception as model_error:
                                            print(f"[ERROR] Step {i}: auto_refine_model failed: {model_error}")
                                            print(f"[DEBUG] frame stats: min={frame.min().item():.6f}, max={frame.max().item():.6f}")
                                            raise RuntimeError(f"auto_refine_model failed at step {i}, frame {t_idx}: {model_error}")
                                        
                                        print(f"[DEBUG] Step {i}: heatmap computed, shape = {heatmap.shape}")
                                        
                                        # NaN/Inf チェック
                                        if torch.isnan(heatmap).any():
                                            print(f"[ERROR] Step {i}: NaN detected in heatmap!")
                                            print(f"[DEBUG] heatmap stats: min={heatmap.min().item():.6f}, max={heatmap.max().item():.6f}, mean={heatmap.mean().item():.6f}")
                                            print(f"[DEBUG] frame stats: min={frame.min().item():.6f}, max={frame.max().item():.6f}, mean={frame.mean().item():.6f}")
                                            # NaN発生時は安全なゼロheatmapで代替
                                            print(f"[WARNING] Step {i}: Replacing NaN heatmap with zeros")
                                            heatmap = torch.zeros_like(heatmap)
                                        if torch.isinf(heatmap).any():
                                            print(f"[ERROR] Step {i}: Inf detected in heatmap!")
                                            print(f"[DEBUG] heatmap stats: min={heatmap.min().item():.6f}, max={heatmap.max().item():.6f}")
                                            # Inf発生時は安全なクランプで修正
                                            print(f"[WARNING] Step {i}: Clamping Inf heatmap values")
                                            heatmap = torch.clamp(heatmap, min=0.0, max=1.0)
                                        
                                        # heatmapの平均計算を安全に実行
                                        heatmap_mean = heatmap.mean()
                                        if torch.isnan(heatmap_mean) or torch.isinf(heatmap_mean):
                                            print(f"[WARNING] Step {i}: Invalid heatmap mean, using zero")
                                            heatmap_mean = torch.zeros_like(heatmap_mean)
                                        
                                        heat_loss = heatmap_mean * heatmap_guidance_strength
                                        
                                        # heat_loss の NaN/Inf チェック
                                        if torch.isnan(heat_loss).any():
                                            print(f"[ERROR] Step {i}: NaN detected in heat_loss!")
                                            print(f"[DEBUG] heat_loss = {heat_loss.item():.6f}")
                                            print(f"[DEBUG] heatmap_guidance_strength = {heatmap_guidance_strength}")
                                            raise RuntimeError(f"NaN detected in heat_loss at step {i}, frame {t_idx}")
                                        if torch.isinf(heat_loss).any():
                                            print(f"[ERROR] Step {i}: Inf detected in heat_loss!")
                                            print(f"[DEBUG] heat_loss = {heat_loss.item():.6f}")
                                            raise RuntimeError(f"Inf detected in heat_loss at step {i}, frame {t_idx}")
                                        
                                        loss = loss + heat_loss / chunk_frames  # チャンク内で平均
                                        
                                        # accumulated loss の NaN/Inf チェック
                                        if torch.isnan(loss).any():
                                            print(f"[ERROR] Step {i}: NaN detected in accumulated loss!")
                                            print(f"[DEBUG] loss = {loss.item():.6f}")
                                            print(f"[DEBUG] heat_loss = {heat_loss.item():.6f}")
                                            print(f"[DEBUG] chunk_frames = {chunk_frames}")
                                            raise RuntimeError(f"NaN detected in accumulated loss at step {i}, frame {t_idx}")
                                        if torch.isinf(loss).any():
                                            print(f"[ERROR] Step {i}: Inf detected in accumulated loss!")
                                            print(f"[DEBUG] loss = {loss.item():.6f}")
                                            raise RuntimeError(f"Inf detected in accumulated loss at step {i}, frame {t_idx}")
                                        
                                        print(f"[DEBUG] Step {i}: Frame {t_idx} processed, accumulated loss = {loss.item():.6f}")
                                
                                # ---- 3) backward → grad 取得 & 蓄積 ----
                                print(f"[DEBUG] Step {i}: Computing gradients - loss.backward()")
                                try:
                                    g, = torch.autograd.grad(loss, latent_chunk, retain_graph=False)
                                    print(f"[DEBUG] Step {i}: Gradients computed successfully, g.shape = {g.shape}")
                                    
                                    # latent → x_t への chain rule: dL/dx_t = (dL/dlatent) * (dlatent/dx_t)
                                    print(f"[DEBUG] Step {i}: Computing chain rule gradients to x_t")
                                    g_xt, = torch.autograd.grad(latent_chunk, x_t,
                                                                grad_outputs=g,
                                                                retain_graph=True)  # グラフは次 chunk でも使う
                                    print(f"[DEBUG] Step {i}: Chain rule gradients computed, g_xt.shape = {g_xt.shape}")
                                    
                                    # x_tの該当部分に勾配を蓄積（latentの時間次元に対応）
                                    if x_t.dim() == 5:  # (B,16,T,H8,W8)
                                        # g_xtをchunkの時間次元にスライスしてから蓄積
                                        g_xt_chunk = g_xt[:, :, start:end, :, :]
                                        grad_buf[:, :, start:end, :, :] += g_xt_chunk.detach()
                                        print(f"[DEBUG] Step {i}: Gradients accumulated to grad_buf for chunk {start}:{end}")
                                    else:  # 4D の場合は全体に適用
                                        grad_buf += g_xt.detach()
                                        print(f"[DEBUG] Step {i}: Gradients accumulated to grad_buf (4D case)")
                                    
                                except Exception as grad_error:
                                    print(f"[DEBUG] Step {i}: ERROR in gradient computation: {grad_error}")
                                    print(f"[DEBUG] Step {i}: loss.requires_grad = {loss.requires_grad}")
                                    print(f"[DEBUG] Step {i}: latent_chunk.requires_grad = {latent_chunk.requires_grad}")
                                    raise
                                
                                # ---- 4) メモリ解放 ----
                                print(f"[DEBUG] Step {i}: Cleaning up chunk memory")
                                del latent_chunk, img_chunk, g, g_xt, loss
                                torch.cuda.empty_cache()
                                print(f"[DEBUG] Step {i}: Chunk {start}:{end} processing completed")
                                
                        elif x0_hat_fp16.dim() == 4:                # (B,16,H8,W8) - 単一フレーム
                            B, C16, H8, W8 = x0_hat_fp16.shape
                            print(f"[DEBUG] Step {i}: Processing single frame")
                            
                            # ---- 1) latent 準備 ----
                            print(f"[DEBUG] Step {i}: Preparing single frame latent")
                            
                            # x0_hat_fp16のNaN/Infチェック（single frame）
                            if torch.isnan(x0_hat_fp16).any():
                                print(f"[ERROR] Step {i}: NaN detected in x0_hat_fp16 (single frame)!")
                                print(f"[DEBUG] x0_hat_fp16 stats: min={x0_hat_fp16.min().item():.6f}, max={x0_hat_fp16.max().item():.6f}")
                                print(f"[WARNING] Step {i}: Skipping single frame processing due to NaN")
                                return torch.zeros_like(x_t)  # 安全なゼロ勾配を返す
                            
                            if torch.isinf(x0_hat_fp16).any():
                                print(f"[ERROR] Step {i}: Inf detected in x0_hat_fp16 (single frame)!")
                                x0_hat_fp16 = torch.clamp(x0_hat_fp16, min=-50.0, max=50.0)
                                print(f"[WARNING] Step {i}: Clamped Inf values in x0_hat_fp16 (single frame)")
                            
                            latent = x0_hat_fp16.unsqueeze(2).clone().cpu()     # (B,16,1,H8,W8) → CPU, clone()追加
                            latent = latent.to(GPU_VAE, non_blocking=True)  # 独立コピー
                            latent.requires_grad_(True)
                            print(f"[DEBUG] Step {i}: Single frame latent prepared, requires_grad = {latent.requires_grad}")
                            print(f"[DEBUG] Step {i}: single frame latent range: [{latent.min().item():.6f}, {latent.max().item():.6f}]")
                            
                            # ---- 2) forward ----
                            with torch.set_grad_enabled(True):
                                img = vae_fp16.decode(latent / 0.18215).sample   # (B,3,1,H,W)
                                img = ((img + 1) / 2).to(x_t.dtype)
                                img = img.squeeze(2)                             # (B,3,H,W)
                                img = img.clone()  # 画像のコピー（inference tensor回避）
                                
                                # ヒートマップ計算（安全な処理）
                                try:
                                    safe_img = torch.clamp(img, min=0.0, max=1.0)
                                    heatmap = auto_refine_model(safe_img)
                                except Exception as model_error:
                                    print(f"[ERROR] Step {i}: auto_refine_model failed (single frame): {model_error}")
                                    print(f"[DEBUG] img stats: min={img.min().item():.6f}, max={img.max().item():.6f}")
                                    raise RuntimeError(f"auto_refine_model failed at step {i} (single frame): {model_error}")
                                
                                # NaN/Inf チェック (single frame case)
                                if torch.isnan(heatmap).any():
                                    print(f"[ERROR] Step {i}: NaN detected in heatmap (single frame)!")
                                    print(f"[DEBUG] heatmap stats: min={heatmap.min().item():.6f}, max={heatmap.max().item():.6f}")
                                    print(f"[DEBUG] img stats: min={img.min().item():.6f}, max={img.max().item():.6f}")
                                    print(f"[WARNING] Step {i}: Replacing NaN heatmap with zeros (single frame)")
                                    heatmap = torch.zeros_like(heatmap)
                                if torch.isinf(heatmap).any():
                                    print(f"[ERROR] Step {i}: Inf detected in heatmap (single frame)!")
                                    print(f"[WARNING] Step {i}: Clamping Inf heatmap values (single frame)")
                                    heatmap = torch.clamp(heatmap, min=-10.0, max=10.0)
                                
                                # 安全なloss計算
                                heatmap_mean = heatmap.mean()
                                if torch.isnan(heatmap_mean) or torch.isinf(heatmap_mean):
                                    print(f"[WARNING] Step {i}: Invalid heatmap mean (single frame), using zero")
                                    heatmap_mean = torch.zeros_like(heatmap_mean)
                                
                                loss = heatmap_mean * heatmap_guidance_strength
                                
                                # loss の NaN/Inf チェック (single frame case)
                                if torch.isnan(loss).any():
                                    print(f"[ERROR] Step {i}: NaN detected in loss (single frame)!")
                                    print(f"[DEBUG] loss = {loss.item():.6f}")
                                    raise RuntimeError(f"NaN detected in loss at step {i} (single frame)")
                                if torch.isinf(loss).any():
                                    print(f"[ERROR] Step {i}: Inf detected in loss (single frame)!")
                                    raise RuntimeError(f"Inf detected in loss at step {i} (single frame)")
                            
                            # ---- 3) backward ----
                            g, = torch.autograd.grad(loss, latent, retain_graph=False)
                            g_xt, = torch.autograd.grad(latent, x_t,
                                                        grad_outputs=g,
                                                        retain_graph=False)
                            grad_buf += g_xt.detach()
                            
                            # ---- 4) メモリ解放 ----
                            del latent, img, g, g_xt, loss
                            torch.cuda.empty_cache()

                        else:
                            raise RuntimeError(f"Unexpected latent dim {x0_hat_fp16.dim()}")

                        torch.cuda.empty_cache()
                        print(f"[DEBUG] Step {i}: Chunk-based VAE decode completed")

                    except Exception as e:
                        print(f"[DEBUG] Step {i}: Error in chunk-based VAE decode: {e}")
                        print(f"[DEBUG] Step {i}: Exception type: {type(e)}")
                        import traceback
                        print(f"[DEBUG] Step {i}: Traceback: {traceback.format_exc()}")
                        raise
                elif degradation_mode == "latent" or (degradation_mode == "image" and img is None):
                    # latent空間モード: x0_hatを直接使用（VAEデコード不要、チャンクベース）
                    try:
                        effective_mode = "latent" if degradation_mode == "latent" else "latent (fallback)"
                        print(f"[DEBUG] Step {i}: Using {effective_mode} mode for heatmap guidance")
                        print(f"[DEBUG] Step {i}: x0_hat shape: {x0_hat.shape}, dim: {x0_hat.dim()}")
                        
                        if x0_hat.dim() == 5:                  # (B,16,T,H8,W8)
                            B, C16, T, H8, W8 = x0_hat.shape
                            chunk_T = min(1, T)  # チャンクサイズ（メモリに応じて調整）
                            print(f"[DEBUG] Step {i}: Processing {T} latent frames in chunks of {chunk_T}")
                            
                            for start in range(0, T, chunk_T):
                                end = min(start + chunk_T, T)
                                print(f"[DEBUG] Step {i}: Processing latent chunk {start}:{end}")
                                
                                # ---- 1) latent チャンク抽出 ----
                                print(f"[DEBUG] Step {i}: Extracting latent chunk {start}:{end}")
                                
                                # x0_hatのNaN/Infチェック（latent mode）
                                chunk_source = x0_hat[:, :, start:end, :, :]
                                if torch.isnan(chunk_source).any():
                                    print(f"[ERROR] Step {i}: NaN detected in latent chunk source!")
                                    print(f"[DEBUG] chunk_source stats: min={chunk_source.min().item():.6f}, max={chunk_source.max().item():.6f}")
                                    print(f"[WARNING] Step {i}: Skipping latent chunk {start}:{end} due to NaN")
                                    continue
                                
                                if torch.isinf(chunk_source).any():
                                    print(f"[ERROR] Step {i}: Inf detected in latent chunk source!")
                                    chunk_source = torch.clamp(chunk_source, min=-50.0, max=50.0)
                                    print(f"[WARNING] Step {i}: Clamped Inf values in latent chunk source")
                                
                                latent_chunk = chunk_source.clone().cpu()        # CPU へ, clone()追加
                                latent_chunk = latent_chunk.float().requires_grad_(True)   # 独立コピー + 勾配有効化
                                print(f"[DEBUG] Step {i}: Latent chunk prepared, requires_grad = {latent_chunk.requires_grad}")
                                print(f"[DEBUG] Step {i}: latent chunk range: [{latent_chunk.min().item():.6f}, {latent_chunk.max().item():.6f}]")
                                
                                # ---- 2) forward (heat-map) ----
                                with torch.set_grad_enabled(True):
                                    # CPUからGPUへ移動（auto_refine_modelと同じデバイス）
                                    latent_chunk = latent_chunk.to(auto_refine_device, non_blocking=True)
                                    
                                    # フレームごと heat-map → loss
                                    loss = torch.zeros(1, device=latent_chunk.device, dtype=x_t.dtype, requires_grad=True)
                                    chunk_frames = latent_chunk.size(2)
                                    
                                    for t_idx in range(chunk_frames):
                                        latent_frame = latent_chunk[:, :, t_idx]  # (B, 16, H8, W8)
                                        
                                        try:
                                            # latent frameを安全な範囲にクランプ
                                            safe_latent_frame = torch.clamp(latent_frame, min=-50.0, max=50.0)
                                            heatmap = auto_refine_model(safe_latent_frame)  # (B, 1, H8, W8)
                                        except Exception as model_error:
                                            print(f"[ERROR] Step {i}: auto_refine_model failed (latent mode): {model_error}")
                                            print(f"[DEBUG] latent_frame stats: min={latent_frame.min().item():.6f}, max={latent_frame.max().item():.6f}")
                                            raise RuntimeError(f"auto_refine_model failed at step {i}, latent frame {t_idx}: {model_error}")
                                        
                                        # NaN/Inf チェック (latent mode)
                                        if torch.isnan(heatmap).any():
                                            print(f"[ERROR] Step {i}: NaN detected in heatmap (latent mode)!")
                                            print(f"[DEBUG] latent_frame stats: min={latent_frame.min().item():.6f}, max={latent_frame.max().item():.6f}")
                                            print(f"[WARNING] Step {i}: Replacing NaN heatmap with zeros (latent mode)")
                                            heatmap = torch.zeros_like(heatmap)
                                        if torch.isinf(heatmap).any():
                                            print(f"[ERROR] Step {i}: Inf detected in heatmap (latent mode)!")
                                            print(f"[WARNING] Step {i}: Clamping Inf heatmap values (latent mode)")
                                            heatmap = torch.clamp(heatmap, min=-10.0, max=10.0)
                                        
                                        # 安全なheat_loss計算
                                        heatmap_mean = heatmap.mean()
                                        if torch.isnan(heatmap_mean) or torch.isinf(heatmap_mean):
                                            print(f"[WARNING] Step {i}: Invalid heatmap mean (latent mode), using zero")
                                            heatmap_mean = torch.zeros_like(heatmap_mean)
                                        
                                        heat_loss = heatmap_mean * heatmap_guidance_strength
                                        
                                        # heat_loss の NaN/Inf チェック (latent mode)
                                        if torch.isnan(heat_loss).any():
                                            print(f"[ERROR] Step {i}: NaN detected in heat_loss (latent mode)!")
                                            print(f"[DEBUG] heat_loss = {heat_loss.item():.6f}")
                                            raise RuntimeError(f"NaN detected in heat_loss at step {i}, latent frame {t_idx}")
                                        if torch.isinf(heat_loss).any():
                                            print(f"[ERROR] Step {i}: Inf detected in heat_loss (latent mode)!")
                                            raise RuntimeError(f"Inf detected in heat_loss at step {i}, latent frame {t_idx}")
                                        
                                        loss = loss + heat_loss / chunk_frames  # チャンク内で平均
                                        
                                        # accumulated loss の NaN/Inf チェック (latent mode)
                                        if torch.isnan(loss).any():
                                            print(f"[ERROR] Step {i}: NaN detected in accumulated loss (latent mode)!")
                                            print(f"[DEBUG] loss = {loss.item():.6f}")
                                            raise RuntimeError(f"NaN detected in accumulated loss at step {i}, latent frame {t_idx}")
                                        if torch.isinf(loss).any():
                                            print(f"[ERROR] Step {i}: Inf detected in accumulated loss (latent mode)!")
                                            raise RuntimeError(f"Inf detected in accumulated loss at step {i}, latent frame {t_idx}")
                                
                                # ---- 3) backward → grad 取得 & 蓄積 ----
                                g, = torch.autograd.grad(loss, latent_chunk, retain_graph=False)
                                
                                # latent → x_t への chain rule
                                g_xt, = torch.autograd.grad(latent_chunk, x_t,
                                                            grad_outputs=g,
                                                            retain_graph=True)
                                
                                # x_tの該当部分に勾配を蓄積（g_xtをchunkサイズにスライス）
                                g_xt_chunk = g_xt[:, :, start:end, :, :]
                                grad_buf[:, :, start:end, :, :] += g_xt_chunk.detach()
                                
                                # ---- 4) メモリ解放 ----
                                del latent_chunk, g, g_xt, loss
                                torch.cuda.empty_cache()
                                
                        elif x0_hat.dim() == 4:                # (B,16,H8,W8) - 単一フレーム
                            B, C16, H8, W8 = x0_hat.shape
                            print(f"[DEBUG] Step {i}: Processing single latent frame")
                            
                            # ---- 1) latent 準備 ----
                            print(f"[DEBUG] Step {i}: Preparing single latent frame")
                            
                            # x0_hatのNaN/Infチェック（single latent frame）
                            if torch.isnan(x0_hat).any():
                                print(f"[ERROR] Step {i}: NaN detected in x0_hat (single latent frame)!")
                                print(f"[DEBUG] x0_hat stats: min={x0_hat.min().item():.6f}, max={x0_hat.max().item():.6f}")
                                print(f"[WARNING] Step {i}: Skipping single latent frame processing due to NaN")
                                return torch.zeros_like(x_t)  # 安全なゼロ勾配を返す
                            
                            if torch.isinf(x0_hat).any():
                                print(f"[ERROR] Step {i}: Inf detected in x0_hat (single latent frame)!")
                                x0_hat = torch.clamp(x0_hat, min=-50.0, max=50.0)
                                print(f"[WARNING] Step {i}: Clamped Inf values in x0_hat (single latent frame)")
                            
                            latent = x0_hat.clone().cpu().float().requires_grad_(True)  # CPU経由で独立コピー, clone()追加
                            print(f"[DEBUG] Step {i}: Single latent frame prepared, requires_grad = {latent.requires_grad}")
                            print(f"[DEBUG] Step {i}: single latent frame range: [{latent.min().item():.6f}, {latent.max().item():.6f}]")
                            
                            # ---- 2) forward ----
                            with torch.set_grad_enabled(True):
                                # CPUからGPUへ移動
                                latent = latent.to(auto_refine_device, non_blocking=True)
                                
                                try:
                                    # latentを安全な範囲にクランプ
                                    safe_latent = torch.clamp(latent, min=-50.0, max=50.0)
                                    heatmap = auto_refine_model(safe_latent)
                                except Exception as model_error:
                                    print(f"[ERROR] Step {i}: auto_refine_model failed (single latent frame): {model_error}")
                                    print(f"[DEBUG] latent stats: min={latent.min().item():.6f}, max={latent.max().item():.6f}")
                                    raise RuntimeError(f"auto_refine_model failed at step {i} (single latent frame): {model_error}")
                                
                                # NaN/Inf チェック (single latent frame)
                                if torch.isnan(heatmap).any():
                                    print(f"[ERROR] Step {i}: NaN detected in heatmap (single latent frame)!")
                                    print(f"[DEBUG] latent stats: min={latent.min().item():.6f}, max={latent.max().item():.6f}")
                                    print(f"[WARNING] Step {i}: Replacing NaN heatmap with zeros (single latent frame)")
                                    heatmap = torch.zeros_like(heatmap)
                                if torch.isinf(heatmap).any():
                                    print(f"[ERROR] Step {i}: Inf detected in heatmap (single latent frame)!")
                                    print(f"[WARNING] Step {i}: Clamping Inf heatmap values (single latent frame)")
                                    heatmap = torch.clamp(heatmap, min=-10.0, max=10.0)
                                
                                # 安全なloss計算
                                heatmap_mean = heatmap.mean()
                                if torch.isnan(heatmap_mean) or torch.isinf(heatmap_mean):
                                    print(f"[WARNING] Step {i}: Invalid heatmap mean (single latent frame), using zero")
                                    heatmap_mean = torch.zeros_like(heatmap_mean)
                                
                                loss = heatmap_mean * heatmap_guidance_strength
                                
                                # loss の NaN/Inf チェック (single latent frame)
                                if torch.isnan(loss).any():
                                    print(f"[ERROR] Step {i}: NaN detected in loss (single latent frame)!")
                                    print(f"[DEBUG] loss = {loss.item():.6f}")
                                    raise RuntimeError(f"NaN detected in loss at step {i} (single latent frame)")
                                if torch.isinf(loss).any():
                                    print(f"[ERROR] Step {i}: Inf detected in loss (single latent frame)!")
                                    raise RuntimeError(f"Inf detected in loss at step {i} (single latent frame)")
                            
                            # ---- 3) backward ----
                            g, = torch.autograd.grad(loss, latent, retain_graph=False)
                            g_xt, = torch.autograd.grad(latent, x_t,
                                                        grad_outputs=g,
                                                        retain_graph=False)
                            grad_buf += g_xt.detach()
                            
                            # ---- 4) メモリ解放 ----
                            del latent, g, g_xt, loss
                            torch.cuda.empty_cache()

                        else:
                            raise RuntimeError(f"Unexpected x0_hat dim {x0_hat.dim()}")
                            
                        print(f"[DEBUG] Step {i}: Chunk-based latent mode completed")
                        
                    except Exception as e:
                        print(f"[DEBUG] Step {i}: Error in chunk-based latent processing: {e}")
                        raise
                else:
                    print(f"[DEBUG] Step {i}: Skipping heatmap guidance (mode={degradation_mode})")
                
                # ---- 最終処理: grad_buf を返す ----
                print(f"[DEBUG] Step {i}: Returning accumulated gradients from chunks")
                
                # ステップ終了時のメモリクリーンアップ
                if 'img' in locals():
                    del img
                if 'eps' in locals():
                    del eps
                if 'x0_hat' in locals():
                    del x0_hat
                torch.cuda.empty_cache()
                
                return -grad_buf  # チャンクベース処理で蓄積された勾配を返す
    
    # エッジ強化またはヒートマップガイダンスが有効な場合のみcond_fnを設定
    cond_fn = guidance_cond_fn if (heatmap_guidance_strength > 0.0) else None

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    # 元のコールバックをそのまま使用（cond_fnで勾配ガイダンスを実装）

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
        print(f"[DEBUG] About to call sample_unipc")
        print(f"[DEBUG] latents dtype: {latents.dtype}, sigmas dtype: {sigmas.dtype}")
        print(f"[DEBUG] cond_fn: {cond_fn}")
        try:
            results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback, cond_fn=cond_fn, cond_scale=heatmap_guidance_scale)
            print(f"[DEBUG] sample_unipc completed successfully")
        except Exception as e:
            print(f"[DEBUG] Error in sample_unipc: {e}")
            raise
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    if heatmap_guidance_strength > 0.0:
        print(f"[DEBUG] Heatmap guidance applied via cond_fn gradient guidance: strength={heatmap_guidance_strength}")

    return results
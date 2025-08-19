# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0
# Attribution-ShareAlike 4.0 International Licence


import torch

from tqdm.auto import trange


def expand_dims(v, dims):
    print(f"[DEBUG] expand_dims called with v.dtype: {v.dtype}, dims: {dims}")
    try:
        result = v[(...,) + (None,) * (dims - 1)]
        print(f"[DEBUG] expand_dims completed, result.dtype: {result.dtype}")
        return result
    except Exception as e:
        print(f"[DEBUG] Error in expand_dims: {e}")
        raise


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant='bh1'):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args

    def model_fn(self, x, t):
        print(f"[DEBUG] model_fn called with x.dtype: {x.dtype}, t.dtype: {t.dtype}")
        try:
            result = self.model(x, t, **self.extra_args)
            print(f"[DEBUG] model call completed, result.dtype: {result.dtype}")
            return result
        except Exception as e:
            print(f"[DEBUG] Error in model call: {e}")
            raise

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        print(f"[DEBUG] update_fn called with order={order}")
        print(f"[DEBUG] x.dtype: {x.dtype}, t.dtype: {t.dtype}")
        assert order <= len(model_prev_list)
        dims = x.dim()

        try:
            t_prev_0 = t_prev_list[-1]
            print(f"[DEBUG] t_prev_0.dtype: {t_prev_0.dtype}")
            lambda_prev_0 = - torch.log(t_prev_0)  # すでにfloat32なのでfloat()は不要
            lambda_t = - torch.log(t)  # すでにfloat32なのでfloat()は不要
            print(f"[DEBUG] lambda calculations completed")
            model_prev_0 = model_prev_list[-1]
            print(f"[DEBUG] model_prev_0.dtype: {model_prev_0.dtype}")

            h = lambda_t - lambda_prev_0
            print(f"[DEBUG] h calculation completed")
        except Exception as e:
            print(f"[DEBUG] Error in update_fn initial calculations: {e}")
            raise

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = - torch.log(t_prev_i)  # すでにfloat32なのでfloat()は不要
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], device=b.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        if order == 1:
            rhos_c = torch.tensor([0.5], device=b.device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        if use_predictor:
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        model_t = self.model_fn(x_t, t)

        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None,
               cond_fn=None, cond_scale=1.0,   # ★ 追加
               disable_pbar=False):
        print(f"[DEBUG] UniPC sample starting, x.dtype: {x.dtype}, sigmas.dtype: {sigmas.dtype}")
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            print(f"[DEBUG] Starting step {i}")
            try:
                vec_t = sigmas[i].expand(x.shape[0])  # dtypeはすでにfloat32なのでそのまま使用
                print(f"[DEBUG] Step {i}: vec_t.dtype: {vec_t.dtype}")
            except Exception as e:
                print(f"[DEBUG] Step {i}: Error in vec_t creation: {e}")
                raise

            try:
                if i == 0:
                    print(f"[DEBUG] Step {i}: Calling model_fn")
                    model_prev_list = [self.model_fn(x, vec_t)]
                    t_prev_list = [vec_t]
                    print(f"[DEBUG] Step {i}: model_fn completed")
                elif i < order:
                    print(f"[DEBUG] Step {i}: Calling update_fn with init_order={i}")
                    init_order = i
                    x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                    print(f"[DEBUG] Step {i}: update_fn completed")
                else:
                    print(f"[DEBUG] Step {i}: Calling update_fn with order={order}")
                    x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                    print(f"[DEBUG] Step {i}: update_fn completed")
            except Exception as e:
                print(f"[DEBUG] Step {i}: Error in step operations: {e}")
                raise

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                print(f"[DEBUG] Step {i}: Calling callback")
                try:
                    callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})
                    print(f"[DEBUG] Step {i}: Callback completed")
                except Exception as e:
                    print(f"[DEBUG] Step {i}: Error in callback: {e}")
                    raise

            # ===== Gradient Guidance (cond_fn) =========================
            if cond_fn is not None:
                print(f"[DEBUG] uni_pc_fm calling cond_fn at step {i}")
                print(f"[DEBUG] x dtype: {x.dtype}, vec_t dtype: {vec_t.dtype}")
                with torch.enable_grad():
                    x_req = x.detach().requires_grad_(True)
                    print(f"[DEBUG] x_req dtype: {x_req.dtype}")
                    # cond_fn は「勾配（∂L/∂x_t）」を直接返す仕様にする
                    try:
                        cond_grad = cond_fn(x_req, vec_t, i)
                        print(f"[DEBUG] cond_fn returned successfully")
                    except Exception as e:
                        print(f"[DEBUG] Error in cond_fn: {e}")
                        raise
                    if cond_grad is None:
                        # 勾配がない場合はスキップ
                        g = torch.zeros_like(x_req)
                    elif torch.is_tensor(cond_grad):
                        # 勾配そのものを返す想定（k‑diffusion と同じ）
                        g = cond_grad
                    else:
                        # あるいはスカラー損失を返しても許容
                        g = torch.autograd.grad(cond_grad, x_req)[0]

                # 勾配がゼロでない場合のみ適用
                if g is not None and not torch.allclose(g, torch.zeros_like(g)):
                    # x ← x - s · σ_t² · g   （s = cond_scale）
                    x = (x_req - cond_scale *
                         (vec_t**2).reshape(-1, *([1]*(x.dim()-1))) * g
                         ).detach()
                else:
                    x = x_req.detach()
            # =============================================
            
            print(f"[DEBUG] Step {i} completed successfully")

        print(f"[DEBUG] UniPC sampling completed successfully")
        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, cond_fn=None, cond_scale=1.0, disable=False, variant='bh1'):
    assert variant in ['bh1', 'bh2']
    
    # 元のdtypeを保存
    original_dtype = noise.dtype
    
    # float32に変換してから処理
    noise_f32 = noise.float()
    sigmas_f32 = sigmas.float()
    
    print(f"[DEBUG] sample_unipc: Converting from {original_dtype} to float32")
    
    # float32で処理
    result = FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(
        noise_f32, sigmas=sigmas_f32, callback=callback, cond_fn=cond_fn, 
        cond_scale=cond_scale, disable_pbar=disable
    )
    
    # 元のdtypeに戻す
    result = result.to(original_dtype)
    print(f"[DEBUG] sample_unipc: Converting result back to {original_dtype}")
    
    return result

# nerf_sigma/train/train_sigma.py
from dataclasses import dataclass
import time
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.amp import autocast, GradScaler
from ..learning_utils.metrics import psnr_from_mse, count_params
from ..learning_utils.sched import make_warmup_cosine_lambda
from ..learning_utils.chunk import forward_chunked_sigma
from ..eval.render import eval_psnr_on_frame
from ..learning_utils.checkpoint import save_checkpoint_full
from ..ops.forward_sigma import nerf_opacity
@dataclass
class TrainCfg:
    steps: int = 10_000
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    amp: bool = True
    print_every: int = 100
    eval_every: int = 500
    device: str = "cuda"
    chunk_rays_train: int = 32768
    chunk_rays_eval: int = 65536
    ckpt_best_path: str | None = None
    ckpt_last_path: str | None = None
    eval_idx: int = 0

def train_sigma_only(
    model: nn.Module,
    get_batch_rays,                 # () -> (C_true, delta, pts)
    H_wc: torch.Tensor,             # (B, 4, 4)
    images: torch.Tensor,           # (B, H, W)
    rng,
    cfg: TrainCfg,
    *,
    # Optional injected states for true resume:
    optim: torch.optim.Optimizer | None = None,
    sched: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: GradScaler | None = None,
    start_step: int = 0,
) -> nn.Module:
    device = cfg.device
    model.to(device).train()

    # Create or reuse optimizer/scheduler/scaler
    if optim is None:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay)
    if sched is None:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=make_warmup_cosine_lambda(cfg.steps, cfg.warmup_steps)
        )
    use_cuda_amp = (cfg.amp and str(device).startswith("cuda"))
    if scaler is None:
        scaler = GradScaler("cuda", enabled=use_cuda_amp)

    best_psnr = -1.0
    start_t = time.time()
    global_step = start_step  # <- continue from where we left off

    eval_idx = int(cfg.eval_idx) % images.shape[0]
    H_eval_row = H_wc[eval_idx]
    img_eval_row = images[eval_idx]

    for step in range(start_step + 1, cfg.steps + 1):
        global_step = step
        C_true, delta, pts = get_batch_rays(debug=False)
        C_true, delta, pts = C_true.to(device), delta.to(device), pts.to(device)

        optim.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_cuda_amp):
            if cfg.chunk_rays_train and pts.shape[0] * pts.shape[1] > cfg.chunk_rays_train:
                sigma = forward_chunked_sigma(model, pts, chunk_rays=cfg.chunk_rays_train)
            else:
                sigma = model(pts)
            C_hat = nerf_opacity(sigma, delta, full_output=False)  # provided by nerflab
            loss = mse_loss(C_hat, C_true)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optim)
        scaler.update()
        sched.step()

        if cfg.print_every and (step % cfg.print_every == 0 or step == (start_step + 1)):
            psnr = psnr_from_mse(loss.item())
            lr_now = sched.get_last_lr()[0]
            print(f"[{step:6d}/{cfg.steps}] train_loss={loss.item():.6f} | train_PSNR={psnr:5.2f} dB | lr={lr_now:.2e} | {time.time()-start_t:.1f}s")

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps):
            mse_val, psnr_val = eval_psnr_on_frame(model, H_eval_row, img_eval_row, rng, device, cfg.chunk_rays_eval)
            if cfg.print_every:
                print(f"  → eval frame {eval_idx}: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB")

            if cfg.ckpt_last_path and (step == cfg.steps):
                save_checkpoint_full(cfg.ckpt_last_path, model, optim, sched, scaler, step, psnr_val, cfg.__dict__)
                if cfg.print_every:
                    print(f"  ✓ saved LAST checkpoint: {cfg.ckpt_last_path}")

            if psnr_val > best_psnr and cfg.ckpt_best_path:
                best_psnr = psnr_val
                save_checkpoint_full(cfg.ckpt_best_path, model, optim, sched, scaler, step, psnr_val, cfg.__dict__)
                if cfg.print_every:
                    print(f"  ✓ saved BEST checkpoint: {cfg.ckpt_best_path} (PSNR={psnr_val:.2f} dB)")

    if cfg.print_every:
        print(f"Done. Params: {count_params(model):,}")
    return model

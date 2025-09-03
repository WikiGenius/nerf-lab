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
    """
    Training hyperparameters and runtime settings.

    Notes
    -----
    • `device` is passed to torch.device; e.g., "cuda", "cuda:0", "cpu".
    • AMP is enabled only when `device.type == 'cuda'` and `amp=True`.
    • `chunk_rays_*` control per-step memory by chunking rays.
    • If `ckpt_best_path` / `ckpt_last_path` are set, checkpoints are saved.
    """
    steps: int = 10_000
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    amp: bool = True
    print_every: int = 100
    eval_every: int = 500
    device: str = "cuda"
    chunk_rays_train: int = 32_768
    chunk_rays_eval: int = 65_536
    ckpt_best_path: str | None = None
    ckpt_last_path: str | None = None
    eval_idx: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

def _move_optim_state_to_device(optim: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Move ONLY optimizer state tensors (e.g., AdamW moments) to `device`.

    Important
    ---------
    • Do NOT replace model parameters with `.to(...)` here; parameters must
      remain the original leaf tensors. The MODEL is moved with `model.to(...)`.
    • We also disable foreach to avoid fused-kernel mixed-device quirks.
    """
    for pg in optim.param_groups:
        pg["foreach"] = False  # pragmatic guard

    for state in optim.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v) and v.device != device:
                state[k] = v.to(device, non_blocking=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_sigma_only(
    model: nn.Module,
    get_batch_rays,                 # () -> (C_true, delta, pts)
    H_wc: torch.Tensor,             # (B, 4, 4)
    images: torch.Tensor,           # (B, H, W)
    rng,                            # torch.Generator (ideally on same device)
    cfg: TrainCfg,
    *,
    # Optional injected states for true resume:
    optim: torch.optim.Optimizer | None = None,
    sched: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: GradScaler | None = None,
    start_step: int = 0,
) -> nn.Module:
    """
    Train a sigma-only NeRF model with robust CUDA/AMP handling and safe resume.

    Design goals
    ------------
    1) **Device hygiene**: model, eval tensors, batches, and optimizer state are
       kept on a single device (usually CUDA).
    2) **AMP correctness**: uses modern `autocast(device_type="cuda")` and
       `GradScaler(device_type="cuda")`.
    3) **Resume safety**: injected optimizers have their *state tensors* migrated
       (no replacing params → no non-leaf `.grad` warnings).
    4) **Memory control**: optional ray chunking for forward pass.
    """
    # ---- Device normalization & model placement --------------------------------
    requested = torch.device(cfg.device)
    model.to(requested).train()

    # Normalize to the model's actual device ("cuda" vs "cuda:0" differences)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = requested  # model without parameters (edge case)

    # Ensure evaluation tensors live on the same device (idempotent)
    H_wc = H_wc.to(device, non_blocking=True)
    images = images.to(device, non_blocking=True)

    # ---- Optimizer / Scheduler / Scaler ----------------------------------------
    if optim is None:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.99),
            weight_decay=cfg.weight_decay,
            foreach=False,   # avoids mixed-device foreach pitfalls
        )
    else:
        # If optimizer is injected (resume), migrate ONLY its state tensors.
        _move_optim_state_to_device(optim, device)

    if sched is None:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=make_warmup_cosine_lambda(cfg.steps, cfg.warmup_steps)
        )

    use_amp = (cfg.amp and device.type == "cuda")
    if scaler is None:
        scaler = GradScaler(device_type="cuda", enabled=use_amp)

    # ---- Eval selection ---------------------------------------------------------
    eval_idx = int(cfg.eval_idx) % images.shape[0]
    H_eval_row = H_wc[eval_idx]
    img_eval_row = images[eval_idx]

    # ---- Training loop ----------------------------------------------------------
    best_psnr = -1.0
    start_t = time.time()

    for step in range(start_step + 1, cfg.steps + 1):
        # === 1) Batch fetch ======================================================
        C_true, delta, pts = get_batch_rays(debug=False)
        C_true = C_true.to(device, non_blocking=True)
        delta  = delta.to(device, non_blocking=True)
        pts    = pts.to(device, non_blocking=True)

        # === 2) Forward & loss ===================================================
        optim.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            # Chunking prevents OOM when B*R grows large
            if cfg.chunk_rays_train and (pts.shape[0] * pts.shape[1] > cfg.chunk_rays_train):
                sigma = forward_chunked_sigma(model, pts, chunk_rays=cfg.chunk_rays_train)
            else:
                sigma = model(pts)

            # nerf_opacity must respect device; assume it uses sigma.device internally
            C_hat = nerf_opacity(sigma, delta, full_output=False)
            loss = mse_loss(C_hat, C_true)

        # === 3) Backward, clip, step ============================================
        scaler.scale(loss).backward()

        # Unscale before clipping when using AMP
        scaler.unscale_(optim)
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        # Guard: single-device sanity (early failure with clear message)
        param_devs = {p.device for p in model.parameters()}
        if param_devs != {device}:
            raise RuntimeError(f"Mixed parameter devices: {param_devs} (expected all on {device})")
        for p, st in optim.state.items():
            for k, v in st.items():
                if torch.is_tensor(v) and v.device != p.device:
                    raise RuntimeError(f"Optimizer state '{k}' on {v.device}, but param on {p.device}")

        scaler.step(optim)
        scaler.update()
        sched.step()

        # === 4) Logging ==========================================================
        if cfg.print_every and (step % cfg.print_every == 0 or step == (start_step + 1)):
            psnr_tr = psnr_from_mse(loss.item())
            lr_now = sched.get_last_lr()[0]
            elapsed = time.time() - start_t
            print(f"[{step:6d}/{cfg.steps}] "
                  f"train_loss={loss.item():.6f} | train_PSNR={psnr_tr:5.2f} dB | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # === 5) Periodic evaluation & checkpoints ================================
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps):
            mse_val, psnr_val = eval_psnr_on_frame(
                model, H_eval_row, img_eval_row, rng, device, cfg.chunk_rays_eval
            )
            if cfg.print_every:
                print(f"  → eval frame {eval_idx}: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB")

            # Save "last" at the very end (optional)
            if cfg.ckpt_last_path and (step == cfg.steps):
                save_checkpoint_full(cfg.ckpt_last_path, model, optim, sched, scaler,
                                     step, psnr_val, cfg.__dict__)
                if cfg.print_every:
                    print(f"  ✓ saved LAST checkpoint: {cfg.ckpt_last_path}")

            # Save "best" on improvement (optional)
            if psnr_val > best_psnr and cfg.ckpt_best_path:
                best_psnr = psnr_val
                save_checkpoint_full(cfg.ckpt_best_path, model, optim, sched, scaler,
                                     step, psnr_val, cfg.__dict__)
                if cfg.print_every:
                    print(f"  ✓ saved BEST checkpoint: {cfg.ckpt_best_path} (PSNR={psnr_val:.2f} dB)")

    if cfg.print_every:
        print(f"Done. Params: {count_params(model):,}")
    return model

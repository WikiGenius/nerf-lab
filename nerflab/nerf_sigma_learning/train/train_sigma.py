# nerf_sigma/train/train_sigma.py
from dataclasses import dataclass
from typing import Literal, Optional
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


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainCfg:
    """
    Training hyperparameters and runtime settings.

    device:
      - "auto"  → cuda if available else cpu
      - "cuda", "cuda:0", "cpu", etc.
    amp:
      - Enabled only when device.type == "cuda"
    amp_dtype:
      - "auto" → bfloat16 when supported else float16
      - "bf16" or "fp16" to force a dtype (CUDA only)
    compile_mode:
      - None (default), or one of {"reduce-overhead", "max-autotune"} if torch.compile is available
    grad_accum_steps:
      - Accumulate this many steps before optimizer.step()
    """
    steps: int = 10_000
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    amp: bool = True
    amp_dtype: Literal["auto", "bf16", "fp16"] = "auto"
    compile_mode: Optional[Literal["reduce-overhead", "max-autotune"]] = None
    grad_accum_steps: int = 1
    print_every: int = 100
    eval_every: int = 500
    device: str = "auto"
    chunk_rays_train: int = 32_768
    chunk_rays_eval: int = 65_536
    ckpt_best_path: str | None = None
    ckpt_last_path: str | None = None
    eval_idx: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(dev_str: str) -> torch.device:
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)

def _choose_amp_dtype(device: torch.device, pref: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if pref == "bf16":
        return torch.bfloat16
    if pref == "fp16":
        return torch.float16
    # auto
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.bfloat16 if supports_bf16 else torch.float16

def inspect_optim_state(optim: torch.optim.Optimizer) -> None:
    """Debug helper: prints any state tensors that don't match parameter device."""
    mismatches = []
    for p, state in optim.state.items():
        p_dev = p.device if isinstance(p, torch.Tensor) else None
        for k, v in state.items():
            if torch.is_tensor(v) and p_dev is not None and v.device != p_dev:
                mismatches.append((k, str(v.device), str(p_dev)))
    if mismatches:
        print("⚠️ Mixed devices in optimizer state:")
        for k, vd, pd in mismatches:
            print(f"  state[{k}] on {vd} but param on {pd}")
    else:
        print("✓ Optimizer state tensors match parameter devices")

def normalize_optim_for_device(
    optim: torch.optim.Optimizer,
    device: torch.device,
    *,
    enable_capturable_on_cuda: bool = True,
) -> None:
    """
    Ensure optimizer param groups & state tensors live on `device`.
    - Coerces 'step' (int or tensor) onto the target device.
    - Disables foreach to avoid mixed-device fused kernels.
    - Sets 'capturable' only when on CUDA (safe no-op if absent).
    """
    for pg in optim.param_groups:
        pg["foreach"] = False
        if enable_capturable_on_cuda and device.type == "cuda":
            pg["capturable"] = True

    for state in optim.state.values():
        # Normalize 'step' first (can be int or CPU tensor)
        if "step" in state:
            s = state["step"]
            if torch.is_tensor(s):
                if s.device != device:
                    state["step"] = s.to(device, non_blocking=True)
            else:  # Python int
                state["step"] = torch.tensor(s, device=device)

        # Move all other tensors
        for k, v in list(state.items()):
            if torch.is_tensor(v) and v.device != device:
                state[k] = v.to(device, non_blocking=True)

def _forward_sigma_safe(model: nn.Module,
                        pts: torch.Tensor,
                        chunk_rays: int | None) -> torch.Tensor:
    if chunk_rays is None or pts.dim() < 3:
        return model(pts)

    if pts.dim() == 3:  # (R, N, 3)
        R = pts.shape[0]
        out_chunks = []
        for start in range(0, R, chunk_rays):
            end = min(R, start + chunk_rays)
            block = pts[start:end]                  # (rblock, N, 3)
            out = model(block)                      # (rblock, N) or (rblock, N, 1) → your model’s convention
            out_chunks.append(out)
        return torch.cat(out_chunks, dim=0)         # concat along R

    elif pts.dim() == 4:  # (B, R, N, 3)
        B, R = pts.shape[:2]
        out_chunks = []
        for start in range(0, R, chunk_rays):
            end = min(R, start + chunk_rays)
            block = pts[:, start:end]               # (B, rblock, N, 3)  ✅ correct axis
            out = model(block)                      # (B, rblock, N)    ← expected
            out_chunks.append(out)
        return torch.cat(out_chunks, dim=1)         # concat along R    ✅

    else:
        # Fallback: treat dim -3 as R
        R = pts.shape[-3]
        out_chunks = []
        for start in range(0, R, chunk_rays):
            end = min(R, start + chunk_rays)
            block = pts.narrow(dim=-3, start=start, length=end - start)
            out = model(block)
            out_chunks.append(out)
        return torch.cat(out_chunks, dim=-3)



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
    Train a sigma-only NeRF model with robust CPU/CUDA handling, mixed precision,
    safe resume, and optional compilation & grad accumulation.
    """
    # ---- Device & model placement ---------------------------------------------
    device_req = resolve_device(cfg.device)
    model.to(device_req).train()

    # Normalize to the model's actual device (covers "cuda" vs "cuda:0")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = device_req

    # Optional: torch.compile for speed (PyTorch ≥ 2.0)
    if cfg.compile_mode:
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e:
            print(f"torch.compile unavailable or failed ({e}); continuing without it.")

    # Place eval tensors on device
    H_wc = H_wc.to(device, non_blocking=True)
    images = images.to(device, non_blocking=True)

    # ---- Optimizer / Scheduler / Scaler ---------------------------------------
    if optim is None:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.99),
            weight_decay=cfg.weight_decay,
            foreach=False,  # avoid mixed-device foreach pitfalls
            # capturable set below on CUDA
        )
    # Harmonize optimizer state with param device
    normalize_optim_for_device(optim, device, enable_capturable_on_cuda=True)

    if sched is None:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=make_warmup_cosine_lambda(cfg.steps, cfg.warmup_steps)
        )

    use_amp = (cfg.amp and device.type == "cuda")
    amp_dtype = _choose_amp_dtype(device, cfg.amp_dtype) if use_amp else None
    if scaler is None:
        scaler = GradScaler(device="cuda", enabled=use_amp)

    # ---- Eval selection --------------------------------------------------------
    eval_idx = int(cfg.eval_idx) % images.shape[0]
    H_eval_row = H_wc[eval_idx]
    img_eval_row = images[eval_idx]

    # ---- Training loop ---------------------------------------------------------
    best_psnr = -1.0
    start_wall = time.time()
    accum = max(1, int(cfg.grad_accum_steps))
    assert accum >= 1, "grad_accum_steps must be >= 1"

    # For adaptive chunking on OOM
    cur_chunk = int(cfg.chunk_rays_train) if cfg.chunk_rays_train else None
    min_chunk = 2048  # floor to prevent degenerately small chunks

    # Optional: improve timing accuracy on CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    for step in range(start_step + 1, cfg.steps + 1):
        # === 1) Batch fetch ======================================================
        C_true, delta, pts = get_batch_rays(debug=False)
        C_true = C_true.to(device, non_blocking=True)
        delta  = delta.to(device, non_blocking=True)
        pts    = pts.to(device, non_blocking=True)
        # Rays processed this step (for logging)
        try:
            rays_this = int(pts.shape[0]) * int(pts.shape[1])
        except Exception:
            rays_this = int(pts.numel() // max(1, pts.shape[-1]))  # rough fallback

        # === 2) Forward & loss ===================================================
        # Zero grad only at accumulation boundary
        if (step - 1) % accum == 0:
            optim.zero_grad(set_to_none=True)

        # Mixed precision (CUDA only). dtype is chosen dynamically (bf16 preferred).
        autocast_kwargs = dict(device_type="cuda", enabled=use_amp)
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype

        # Adaptive OOM handling: try with current chunk, shrink if OOM
        while True:
            try:
                with autocast(**autocast_kwargs):
                    if cur_chunk is not None:
                        sigma = _forward_sigma_safe(model, pts, cur_chunk)
                    else:
                        sigma = model(pts)
                        
                    # after computing sigma
                    if sigma.shape[:2] != delta.shape[:2]:
                        raise RuntimeError(
                            f"Shape mismatch before nerf_opacity: sigma{tuple(sigma.shape)} vs delta{tuple(delta.shape)}; "
                            f"expected (B,R,...) alignment on first two dims."
                        )

                    C_hat = nerf_opacity(sigma, delta, full_output=False)
                    loss = mse_loss(C_hat, C_true) / accum  # scale for grad-accum
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg or "cuda error: out of memory" in msg) and cur_chunk:
                    # Reduce chunk and retry
                    new_chunk = max(min_chunk, cur_chunk // 2)
                    if new_chunk == cur_chunk:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if cur_chunk != new_chunk and (step == start_step + 1 or step % cfg.print_every == 0):
                        print(f"OOM caught → reducing chunk_rays_train: {cur_chunk} → {new_chunk}")
                    cur_chunk = new_chunk
                    continue
                raise  # re-raise non-OOM errors

        # === 3) Backward, clip, step ============================================
        scaler.scale(loss).backward()

        # Unscale before clipping when using AMP
        scaler.unscale_(optim)
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        # Guard: single-device sanity (only enforce floats; ints like 'step' can differ)
        param_devs = {p.device for p in model.parameters()}
        if param_devs != {device}:
            raise RuntimeError(f"Mixed parameter devices: {param_devs} (expected all on {device})")
        for p, st in optim.state.items():
            p_dev = p.device
            for k, v in st.items():
                if torch.is_tensor(v) and torch.is_floating_point(v) and v.device != p_dev:
                    raise RuntimeError(f"Optimizer state '{k}' on {v.device}, but param on {p_dev}")

        # Optimizer step only on accumulation boundary
        do_step = (step % accum == 0) or (step == cfg.steps)
        if do_step:
            scaler.step(optim)
            scaler.update()
            sched.step()

        # === 4) Logging ==========================================================
        if cfg.print_every and (step % cfg.print_every == 0 or step == (start_step + 1)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_wall
            psnr_tr = psnr_from_mse((loss.detach() * accum).item())  # unscale for display
            lr_now = sched.get_last_lr()[0]
            mem = ""
            if device.type == "cuda":
                try:
                    ma = torch.cuda.max_memory_allocated(device)
                    mem = f" | max_mem={ma/1e9:.2f} GB"
                except Exception:
                    pass
            print(f"[{step:6d}/{cfg.steps}] "
                  f"loss={((loss*accum).item()):.6f} | PSNR={psnr_tr:5.2f} dB | "
                  f"lr={lr_now:.2e} | rays={rays_this:,} | {elapsed:.1f}s{mem}")

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

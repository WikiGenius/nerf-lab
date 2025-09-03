import torch
from torch import nn
from typing import Optional

def load_model_weights(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[load_model_weights] missing={missing}, unexpected={unexpected}")
    model.to(device)
    print(f"Loaded weights from {ckpt_path} (step={ckpt.get('step')}, psnr={ckpt.get('psnr')})")
    return ckpt

def save_checkpoint_full(path: str, model, optim, sched, scaler, step: int, psnr: float, cfg_dict: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict() if optim is not None else None,
            "sched": sched.state_dict() if sched is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "psnr": psnr,
            "cfg": cfg_dict,
        },
        path,
    )

def load_checkpoint_full(path: str, model, optim, sched, scaler, device: str):
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[load_checkpoint_full] missing={missing}, unexpected={unexpected}")
    if ckpt.get("optim") and optim is not None:
        optim.load_state_dict(ckpt["optim"])
    if ckpt.get("sched") and sched is not None:
        sched.load_state_dict(ckpt["sched"])
    if ckpt.get("scaler") and scaler is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[load_checkpoint_full] scaler load skipped: {e}")
    print(f"Resumed from {path} (step={ckpt.get('step')}, psnr={ckpt.get('psnr')})")
    return ckpt

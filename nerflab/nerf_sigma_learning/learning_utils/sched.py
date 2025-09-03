import math

def make_warmup_cosine_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

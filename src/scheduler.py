import math


# cosine annealing where after cosine decay it is minimum lr
def get_cosine_annealing_lr(step: int, warmup_it: int, cosine_it: int, max_lr: float, min_lr: float):
    assert 0 <= warmup_it < cosine_it
    if step < warmup_it:
        return ((step + 1) / warmup_it) * max_lr
    if warmup_it <= step < cosine_it:
        coeff = (step - warmup_it)/(cosine_it - warmup_it)
        return min_lr + 0.5*(max_lr - min_lr) * (1.0 + math.cos((coeff) * math.pi)) 
    return min_lr



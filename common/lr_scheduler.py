import math
from collections import deque
import torch
import numpy as np

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

    def get(self):
        return self.value

class AdaptiveLearningRateScheduler:
    def __init__(self, start_step=0, max_lr=1e-5, min_lr=None, warmup_steps=100, max_steps=10000, ema_decay=0.9, ema_slow_decay=0.999, increase_factor=1.2, decrease_factor=0.9):
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr * 0.9
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        self.step = start_step
        self.base_lr_ema = EMA(ema_slow_decay)
        self.loss_ema = EMA(ema_decay)
        self.grad_norm_ema = EMA(ema_decay)

        self.loss_history = deque(maxlen=10)
        self.grad_norm_history = deque(maxlen=10)
        self.std_loss_diff_history = deque(maxlen=100)

        self.boost_factor = 1.0

    def freeze_adjust(self, status):
        self.freeze_boost = status

    def inform(self, loss, grad_norm):
        # Ensure the loss and grad_norm are moved to the CPU and converted to float
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().item()
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().cpu().item()
        
        self.loss_ema.update(loss)
        self.grad_norm_ema.update(grad_norm)

        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

    def next_step(self):
        self.step += 1
        if self.step > self.warmup_steps:
            self.base_lr_ema.update(self.calculate_base_lr())
        self.update_boost_factor()

    def get_lr(self):
        base_lr = self.calculate_base_lr()
        if self.step > self.warmup_steps:
            base_lr_ema = self.base_lr_ema.get()
            if base_lr_ema!=None:
                base_lr = base_lr_ema
            else:
                self.base_lr_ema.update(base_lr)

        return base_lr * self.boost_factor

    def calculate_base_lr(self):
        if self.step < self.warmup_steps:
            return self.max_lr * (self.step + 1) / self.warmup_steps

        if self.step > self.max_steps:
            return self.min_lr

        decay_ratio = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0

        lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        
        return lr

    def update_boost_factor(self):
        if self.step <= self.warmup_steps or self.freeze_boost:
            self.boost_factor = 1.0
            print("boost freezed")
            return

        if self.should_increase_boost_factor():
            self.boost_factor *= self.increase_factor
            print("Increase conditions satisfied")
        elif self.should_decrease_boost_factor():
            self.boost_factor *= self.decrease_factor
            print("Decrease conditions satisfied")

        print("Boost constant")
        # Ensure boost factor stays within <0.1, 10> range
        self.boost_factor = max(0.01, min(self.boost_factor, 100))
        print(f"Current boost factor {self.boost_factor}")

    def should_increase_boost_factor(self):
        if len(self.loss_history) < 4:
            return False

        # Relative loss improvement
        loss_improvement = (self.loss_history[-2] - self.loss_history[-1]) / self.loss_history[-1] if self.loss_history[-1] != 0 else 0
        loss_3_steps_back_improved = self.loss_history[-1] < self.loss_history[-4]

        return (
            loss_improvement < 0.03 and  # Loss improved less than 3%
            loss_3_steps_back_improved and  # Loss improved compared to 3 steps back
            self.boost_factor < self.compute_max_allowed_boost_factor()  # Boost factor is less than max allowed
        )

    def should_decrease_boost_factor(self):
        
        recent_losses = list(self.loss_history)[-10:]
        crossed_ema_count = sum(1 for loss in recent_losses if loss > self.loss_ema.get())
        return (self.loss_history[-1] > self.loss_ema.get() and
            crossed_ema_count >= 2) or (
            self.boost_factor > self.compute_max_allowed_boost_factor())  # Boost factor exceeds max allowed
    
    def compute_max_allowed_boost_factor(self):
        if len(self.loss_history) < 10:
            return 1.0
        
        # Calculate differences between consecutive losses
        loss_diffs = np.diff(np.array(self.loss_history))
        
        # Calculate the standard deviation of the differences
        std_loss_diff = np.std(loss_diffs)

        # Normalize the standard deviation
        normalized_std_loss_diff = std_loss_diff / (np.mean(np.abs(loss_diffs)) + 1e-8)
        self.std_loss_diff_history.append(normalized_std_loss_diff)

        # Calculate the current percentile rank of the normalized standard deviation
        if len(self.std_loss_diff_history) < 20:
            return 5  # Not enough history to determine percentiles accurately

        percentile = np.percentile(self.std_loss_diff_history, 95)
        rank = (np.sum(np.array(self.std_loss_diff_history) <= normalized_std_loss_diff) / len(self.std_loss_diff_history))
        rank = 1 - rank

        # Scale max_allowed_boost_factor from 1 to 100 based on the percentile rank
        max_allowed_boost_factor = 1 + rank * 99
        
        print(f"Normalized std of loss differences: {normalized_std_loss_diff}, Percentile rank: {rank}, Max allowed boost factor: {max_allowed_boost_factor}")
        
        return max_allowed_boost_factor

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

    def get(self):
        return self.value

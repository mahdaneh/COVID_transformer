import torch
import math
class WrmUpCosinScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs,base_lr, lr_min=1e-10, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.lr_min = lr_min
        super(WrmUpCosinScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:
            lr = float(self.base_lr *float((self.last_epoch+1) /  self.warmup_epochs))

        else:
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(progress * math.pi))

        return [torch.tensor(lr)]
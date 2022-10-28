from contextlib import contextmanager

import torch

from xlmr_colbert.utils.utils import NullContextManager


class MixedPrecisionManager:
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

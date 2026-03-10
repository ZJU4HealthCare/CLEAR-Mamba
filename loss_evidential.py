import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, kl_coef=1e-3,
                 adaptive=True, c=1.2, kl_min=1e-6, kl_max=2e-2, ema=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.kl_coef = kl_coef

        self.last_nll = None
        self.last_kl = None

        self.adaptive = adaptive
        self.c = c
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.ema = ema
        self.nll_ma = None
        self.kl_ma = None

    def forward(self, evidence, target):
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()

        nll = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        kl_div = self._kl_divergence(alpha)

        nll_mean = nll.mean().item()
        kl_mean  = kl_div.mean().item()
        self.last_nll = nll_mean
        self.last_kl  = kl_mean
        if self.adaptive:

            if self.nll_ma is None:
                self.nll_ma = nll_mean
                self.kl_ma  = max(kl_mean, 1e-8)


            self.nll_ma = self.ema * self.nll_ma + (1.0 - self.ema) * nll_mean
            self.kl_ma  = self.ema * self.kl_ma  + (1.0 - self.ema) * max(kl_mean, 1e-8)

            ratio = self.nll_ma / (self.kl_ma + 1e-8)
            λ_new = max(self.kl_min, min(self.c * ratio, self.kl_max))

            self.kl_coef = self.ema * self.kl_coef + (1.0 - self.ema) * float(λ_new)

        loss = nll + self.kl_coef * kl_div
        return loss.mean()

    def _kl_divergence(self, alpha):
        K = alpha.shape[1]
        S = torch.sum(alpha, dim=1, keepdim=True)
        kl = (
            torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
            + torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)
        )
        return kl.squeeze(1)

def compute_uncertainty(evidence, eps=1e-8):

    alpha = torch.clamp(evidence + 1, min=1e-3) 
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S


    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

    digamma_alpha = torch.digamma(alpha)
    digamma_S = torch.digamma(S)
    mutual_info = torch.sum(probs * (digamma_alpha - digamma_S), dim=1)

    mutual_info = torch.clamp(mutual_info, min=0.0)

    total_uncertainty = entropy + mutual_info
    return entropy, mutual_info, total_uncertainty

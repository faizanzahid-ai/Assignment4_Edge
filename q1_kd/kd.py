"""
Knowledge-distillation framework (Q1.3).

Total student loss
    L = (1 - alpha) * CE(z_s, y)                                  hard-label term
      + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )  soft-target term (Hinton et al.)
      + beta  * sum_i  || phi_i(F_s^i) - F_t^i ||^2               feature (hint) term (FitNets-style)

* Temperature T > 1 softens both distributions so the student also learns the
  "dark knowledge" in the relative probabilities of wrong classes; the T^2
  factor keeps the gradient magnitude of the soft term comparable to CE.
* phi_i is a learnable 1x1 conv + BN "connector" that maps the student feature
  map i to the channel count of the matching teacher feature map; spatial sizes
  are aligned with adaptive average pooling.  Both maps are L2-normalised per
  sample, so the hint loss is scale-free and lies in [0, 4].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_with_temperature(logits, T):
    """Temperature-scaled softmax: exp(z_i / T) / sum_j exp(z_j / T)."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """KL(p_teacher^T || p_student^T) * T^2."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_with_temperature(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureConnector(nn.Module):
    """Projects a student feature map into the teacher's feature space."""

    def __init__(self, s_channels, t_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(s_channels, t_channels, 1, bias=False),
                                  nn.BatchNorm2d(t_channels))

    def forward(self, f_s, target_hw):
        f_s = self.proj(f_s)
        if f_s.shape[-2:] != target_hw:
            f_s = F.adaptive_avg_pool2d(f_s, target_hw)
        return f_s


def hint_loss(f_s, f_t):
    """Squared distance between per-sample L2-normalised feature maps."""
    f_s = F.normalize(f_s.flatten(1).float(), dim=1)
    f_t = F.normalize(f_t.flatten(1).float(), dim=1)
    return (f_s - f_t).pow(2).sum(1).mean()


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-label + feature-based distillation loss."""

    def __init__(self, student_channels, teacher_channels, T=4.0, alpha=0.7, beta=1.0,
                 use_soft=True, use_features=True):
        super().__init__()
        self.T, self.alpha, self.beta = T, alpha, beta
        self.use_soft, self.use_features = use_soft, use_features
        self.connectors = nn.ModuleList(
            [FeatureConnector(s, t) for s, t in zip(student_channels, teacher_channels)])
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, s_feats, t_logits, t_feats, targets):
        parts = {}
        hard = self.ce(s_logits, targets)
        parts["hard"] = hard
        alpha = self.alpha if self.use_soft else 0.0
        loss = (1.0 - alpha) * hard
        if self.use_soft:
            soft = soft_target_loss(s_logits.float(), t_logits.float(), self.T)
            parts["soft"] = soft
            loss = loss + alpha * soft
        if self.use_features and self.beta > 0:
            feat = sum(hint_loss(conn(fs, ft.shape[-2:]), ft)
                       for conn, fs, ft in zip(self.connectors, s_feats, t_feats))
            parts["feat"] = feat
            loss = loss + self.beta * feat
        parts["total"] = loss
        return loss, parts

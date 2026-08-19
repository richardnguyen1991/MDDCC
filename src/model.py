"""MDDCC: 4 nhanh CNN doc lap + compose + FC + Softmax - muc 2.B, 2.C.

Kien truc theo Table 3 cua bai bao, moi subband (cD1, cD2, cD3, cA3) di vao MOT
CNN rieng, KHONG chia se trong so:

    Conv2d(1,  32, 3x3, pad=1) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
    Conv2d(32, 64, 3x3, pad=1) -> ReLU -> MaxPool(2x2) -> Dropout(0.3)
    Conv2d(64, 32, 3x3, pad=1) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)

Compose theo cong thuc (10): z = sum(z_i), flatten -> Linear -> Softmax.
Loss theo muc 2.C: MSE(softmax, one-hot) + lambda_std * sum(sigma(w)).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.nn as nn


# ------------------------------------------------------------------ nhanh
class Branch(nn.Module):
    """Mot nhanh CNN cho mot subband. Table 3."""

    def __init__(self, specs: list[dict], *, in_channels: int = 1,
                 ceil_mode: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for s in specs:
            layers += [
                nn.Conv2d(c_in, s["conv_out"], s["kernel"], padding=s["padding"]),
                nn.ReLU(),
                nn.MaxPool2d(s["pool"], ceil_mode=ceil_mode),
                nn.Dropout(s["dropout"]),
            ]
            c_in = s["conv_out"]
        self.out_channels = c_in
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MDDCC(nn.Module):
    """Multi-Dimensional Deep Convolutional Classifier."""

    def __init__(self, *, num_classes: int, side: int, n_branches: int = 4,
                 branch_specs: list[dict] | None = None, compose: str = "sum",
                 pool_ceil_mode: bool = True):
        super().__init__()
        if compose not in ("sum", "concat"):
            raise ValueError(f"compose phai la 'sum' hoac 'concat', nhan {compose!r}")

        specs = branch_specs or [
            {"conv_out": 32, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.2},
            {"conv_out": 64, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.3},
            {"conv_out": 32, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.2},
        ]
        self.n_branches = n_branches
        self.compose = compose
        self.side = side
        self.num_classes = num_classes
        self.branch_specs = specs

        # KHONG chia se trong so: moi nhanh la mot ModuleList entry rieng
        self.branches = nn.ModuleList(
            [Branch(specs, ceil_mode=pool_ceil_mode) for _ in range(n_branches)])

        with torch.no_grad():
            probe = self.branches[0](torch.zeros(1, 1, side, side))
        c, h, w = probe.shape[1:]
        self.feature_map_shape = (int(c), int(h), int(w))
        flat = c * h * w * (n_branches if compose == "concat" else 1)
        self.flatten_dim = int(flat)
        self.fc = nn.Linear(self.flatten_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_branches, S, S] -> vector da compose [B, flatten_dim]."""
        if x.shape[1] != self.n_branches:
            raise ValueError(
                f"can {self.n_branches} subband, nhan {x.shape[1]}")
        outs = [self.branches[i](x[:, i:i + 1]) for i in range(self.n_branches)]
        if self.compose == "sum":
            z = torch.stack(outs, dim=0).sum(dim=0)   # cong thuc (10)
        else:
            z = torch.cat(outs, dim=1)
        return z.flatten(1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tra ve XAC SUAT softmax - bai bao dung MSE tren dau ra softmax."""
        return torch.softmax(self.logits(x), dim=1)

    def forward_with_branch_mask(self, x: torch.Tensor,
                                 mask: list[bool]) -> torch.Tensor:
        """Zero-out mot so nhanh - phuc vu branch ablation (hinh C11)."""
        if len(mask) != self.n_branches:
            raise ValueError(f"mask phai dai {self.n_branches}")
        outs = []
        for i in range(self.n_branches):
            o = self.branches[i](x[:, i:i + 1])
            outs.append(o if mask[i] else torch.zeros_like(o))
        z = (torch.stack(outs, dim=0).sum(dim=0) if self.compose == "sum"
             else torch.cat(outs, dim=1))
        return torch.softmax(self.fc(z.flatten(1)), dim=1)

    # -------------------------------------------------------------- metadata
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6

    def spec(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "side": self.side,
            "n_branches": self.n_branches,
            "branch_specs": self.branch_specs,
            "compose": self.compose,
            "feature_map_shape": list(self.feature_map_shape),
            "flatten_dim": self.flatten_dim,
            "n_parameters": self.n_parameters(),
            "model_size_mb": round(self.size_mb(), 4),
            "output_activation": "softmax",
        }

    def params_hash(self) -> str:
        """Hash kien truc - checkpoint lech kien truc phai fail-fast (muc 4.3)."""
        payload = {k: v for k, v in self.spec().items()
                   if k not in ("n_parameters", "model_size_mb")}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def build_model(cfg: dict, *, side: int, num_classes: int) -> MDDCC:
    m = cfg["model"]
    return MDDCC(
        num_classes=num_classes,
        side=side,
        n_branches=m.get("num_branches", 4),
        branch_specs=m["branches"],
        compose=m.get("compose", "sum"),
        pool_ceil_mode=m.get("pool_ceil_mode", True),
    )


# ------------------------------------------------------- loss (muc 2.C)
def sigma_of_weight(w: torch.Tensor) -> torch.Tensor:
    """Cong thuc (8): sigma(w) = sqrt( mean(w^2) - mean(w)^2 ).

    Tinh tren ma tran trong so cua MOT lop, KHONG tinh bias.
    Dung dinh nghia cua bai bao chu khong goi torch.std (torch.std mac dinh
    dung ddof=1, se cho gia tri khac).
    """
    flat = w.reshape(-1)
    mean_sq = (flat * flat).mean()
    sq_mean = flat.mean() ** 2
    return torch.sqrt(torch.clamp(mean_sq - sq_mean, min=0.0) + 1e-12)


def std_regularizer(model: nn.Module, *, include_bias: bool = False,
                    layer_types: tuple = (nn.Conv2d, nn.Linear)) -> torch.Tensor:
    """Cong thuc (9): tong sigma(w) qua tat ca lop conv/fc."""
    total = None
    for module in model.modules():
        if not isinstance(module, layer_types):
            continue
        s = sigma_of_weight(module.weight)
        if include_bias and module.bias is not None:
            s = s + sigma_of_weight(module.bias)
        total = s if total is None else total + s
    if total is None:
        raise RuntimeError("Khong tim thay lop conv/linear nao de tinh sigma(w)")
    return total


def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(targets.shape[0], num_classes,
                       dtype=torch.float32, device=targets.device
                       ).scatter_(1, targets.view(-1, 1), 1.0)


def mse_loss(probs: torch.Tensor, targets: torch.Tensor, num_classes: int,
             *, reduction: str = "mean_elements") -> torch.Tensor:
    """MSE giua dau ra softmax va nhan one-hot (muc 2.C).

    reduction:
      mean_elements        - trung binh tren ca batch VA lop (nn.MSELoss mac dinh)
      mean_batch_sum_class - tong theo lop, trung binh theo batch (lon hon
                             num_classes lan; lam MSE khong bi sigma(w) lan at)
    """
    y = one_hot(targets, num_classes)
    se = (probs - y) ** 2
    if reduction == "mean_elements":
        return se.mean()
    if reduction == "mean_batch_sum_class":
        return se.sum(dim=1).mean()
    raise ValueError(f"reduction khong hop le: {reduction!r}")


@dataclass
class LossParts:
    total: torch.Tensor
    mse: torch.Tensor
    std_reg: torch.Tensor


class MDDCCLoss:
    """L = MSE(X, y; w) + lambda_std * sum_layers sigma(w) - cong thuc (9)."""

    def __init__(self, cfg: dict, num_classes: int):
        lcfg = cfg["loss"]
        if lcfg["name"] != "mse":
            raise ValueError(
                f"loss.name={lcfg['name']!r} - muc 2.C bat buoc 'mse' cho run chinh.")
        reg = lcfg.get("std_regularizer", {})
        self.num_classes = num_classes
        self.enabled = reg.get("enabled", True)
        self.lambda_std = float(reg.get("lambda_std", 1.0))
        self.include_bias = reg.get("include_bias", False)
        self.reduction = lcfg.get("mse_reduction", "mean_elements")

    def __call__(self, probs: torch.Tensor, targets: torch.Tensor,
                 model: nn.Module) -> LossParts:
        mse = mse_loss(probs, targets, self.num_classes, reduction=self.reduction)
        if self.enabled and self.lambda_std != 0.0:
            reg = std_regularizer(model, include_bias=self.include_bias)
        else:
            reg = torch.zeros((), device=probs.device)
        return LossParts(mse + self.lambda_std * reg, mse, reg)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"].lower() != "sgd":
        raise ValueError(
            f"optim.name={o['name']!r} - muc 2.D bat buoc SGD cho run chinh.")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(o["learning_rate"]),
        momentum=float(o.get("momentum", 0.0)),
        nesterov=bool(o.get("nesterov", False)),
        weight_decay=float(o.get("weight_decay", 0.0)),
    )


def grad_norm(model: nn.Module) -> float:
    """L2 norm toan cuc cua gradient - panel (d) cua hinh C1."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5

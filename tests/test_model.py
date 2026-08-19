"""Kiem chung kien truc MDDCC va loss - muc 2.B, 2.C."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import model as M

CFG = {
    "model": {
        "num_branches": 4,
        "branches": [
            {"conv_out": 32, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.2},
            {"conv_out": 64, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.3},
            {"conv_out": 32, "kernel": 3, "padding": 1, "pool": 2, "dropout": 0.2},
        ],
        "compose": "sum",
        "pool_ceil_mode": True,
    },
    "loss": {"name": "mse", "mse_reduction": "mean_batch_sum_class",
             "std_regularizer": {"enabled": True, "lambda_std": 1.0,
                                 "include_bias": False}},
    "optim": {"name": "sgd", "learning_rate": 0.001, "momentum": 0.0,
              "nesterov": False, "weight_decay": 0.0},
}


# --------------------------------------------------------------- kien truc
def test_architecture_matches_table3():
    m = M.build_model(CFG, side=10, num_classes=18)
    convs = [x for x in m.branches[0].modules() if isinstance(x, nn.Conv2d)]
    drops = [x for x in m.branches[0].modules() if isinstance(x, nn.Dropout)]
    pools = [x for x in m.branches[0].modules() if isinstance(x, nn.MaxPool2d)]

    assert [c.out_channels for c in convs] == [32, 64, 32]
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs)
    assert [round(d.p, 2) for d in drops] == [0.2, 0.3, 0.2]
    assert len(pools) == 3 and all(p.ceil_mode for p in pools)
    assert any(isinstance(x, nn.ReLU) for x in m.branches[0].modules())


def test_four_independent_branches_do_not_share_weights():
    m = M.build_model(CFG, side=10, num_classes=18)
    assert len(m.branches) == 4
    ids = [id(b.net[0].weight) for b in m.branches]
    assert len(set(ids)) == 4, "4 nhanh phai co trong so RIENG (muc 2.B)"

    w = [b.net[0].weight.detach().clone() for b in m.branches]
    for i in range(1, 4):
        assert not torch.allclose(w[0], w[i]), "trong so khoi tao phai khac nhau"


def test_forward_shape_and_softmax():
    m = M.build_model(CFG, side=10, num_classes=18)
    m.eval()
    out = m(torch.rand(8, 4, 10, 10))
    assert out.shape == (8, 18)
    assert torch.allclose(out.sum(dim=1), torch.ones(8), atol=1e-5)
    assert (out >= 0).all()


def test_compose_sum_is_elementwise_addition():
    """Cong thuc (10): z = sum(z_i), khong phai concat."""
    m = M.build_model(CFG, side=10, num_classes=18)
    m.eval()
    x = torch.rand(4, 4, 10, 10)
    with torch.no_grad():
        manual = sum(m.branches[i](x[:, i:i + 1]) for i in range(4)).flatten(1)
        assert torch.allclose(m.features(x), manual, atol=1e-6)


def test_compose_concat_changes_flatten_dim():
    cfg = {**CFG, "model": {**CFG["model"], "compose": "concat"}}
    s = M.build_model(CFG, side=10, num_classes=18)
    c = M.build_model(cfg, side=10, num_classes=18)
    assert c.flatten_dim == 4 * s.flatten_dim


def test_invalid_compose_rejected():
    cfg = {**CFG, "model": {**CFG["model"], "compose": "mean"}}
    with pytest.raises(ValueError, match="compose"):
        M.build_model(cfg, side=10, num_classes=18)


def test_wrong_subband_count_fails_fast():
    m = M.build_model(CFG, side=10, num_classes=18)
    with pytest.raises(ValueError, match="subband"):
        m(torch.rand(2, 3, 10, 10))


def test_flatten_dim_for_real_geometry():
    """F=81 -> S=10 -> map 2x2 -> 128 chieu (khong phai 32)."""
    m = M.build_model(CFG, side=10, num_classes=18)
    assert m.feature_map_shape == (32, 2, 2)
    assert m.flatten_dim == 128


def test_params_hash_detects_architecture_change():
    a = M.build_model(CFG, side=10, num_classes=18)
    b = M.build_model(CFG, side=10, num_classes=18)
    c = M.build_model(CFG, side=10, num_classes=19)
    assert a.params_hash() == b.params_hash(), "cung kien truc -> cung hash"
    assert a.params_hash() != c.params_hash(), "doi so lop -> doi hash"


def test_branch_mask_zeroes_out_a_branch():
    m = M.build_model(CFG, side=10, num_classes=18)
    m.eval()
    x = torch.rand(4, 4, 10, 10)
    with torch.no_grad():
        full = m.forward_with_branch_mask(x, [True] * 4)
        assert torch.allclose(full, m(x), atol=1e-6)
        ablated = m.forward_with_branch_mask(x, [False, True, True, True])
    assert not torch.allclose(full, ablated), "tat mot nhanh phai doi dau ra"


# -------------------------------------------------------------- sigma(w)
def test_sigma_matches_population_std_formula():
    """Cong thuc (8): sqrt(mean(w^2) - mean(w)^2) = do lech chuan tong the (ddof=0)."""
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    got = float(M.sigma_of_weight(w))
    assert got == pytest.approx(float(np.std(w.numpy(), ddof=0)), abs=1e-6)
    # KHONG phai torch.std mac dinh (ddof=1)
    assert got != pytest.approx(float(torch.std(w)), abs=1e-6)


def test_sigma_of_constant_weight_is_zero():
    assert float(M.sigma_of_weight(torch.full((10,), 3.0))) == pytest.approx(0.0, abs=1e-5)


def test_std_regularizer_sums_over_all_conv_and_fc_layers():
    m = M.build_model(CFG, side=10, num_classes=18)
    total = float(M.std_regularizer(m))
    manual = sum(float(M.sigma_of_weight(mod.weight)) for mod in m.modules()
                 if isinstance(mod, (nn.Conv2d, nn.Linear)))
    assert total == pytest.approx(manual, rel=1e-5)
    # 4 nhanh x 3 conv + 1 fc = 13 lop
    n_layers = sum(1 for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear)))
    assert n_layers == 13


def test_std_regularizer_excludes_bias_by_default():
    m = M.build_model(CFG, side=10, num_classes=18)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.bias.mul_(100.0)
    # doi bias rat manh nhung sigma khong doi vi chi tinh tren weight
    assert float(M.std_regularizer(m)) == pytest.approx(
        float(M.std_regularizer(m, include_bias=False)), rel=1e-9)


def test_std_regularizer_is_differentiable():
    m = M.build_model(CFG, side=10, num_classes=18)
    M.std_regularizer(m).backward()
    assert m.branches[0].net[0].weight.grad is not None
    assert torch.isfinite(m.branches[0].net[0].weight.grad).all()


# ------------------------------------------------------------------ loss
def test_mse_reductions_differ_by_num_classes_factor():
    probs = torch.full((4, 18), 1 / 18)
    y = torch.zeros(4, dtype=torch.long)
    a = float(M.mse_loss(probs, y, 18, reduction="mean_elements"))
    b = float(M.mse_loss(probs, y, 18, reduction="mean_batch_sum_class"))
    assert b == pytest.approx(a * 18, rel=1e-5)


def test_mse_is_zero_for_perfect_prediction():
    y = torch.tensor([0, 1, 2])
    probs = M.one_hot(y, 3)
    assert float(M.mse_loss(probs, y, 3)) == pytest.approx(0.0, abs=1e-9)


def test_loss_splits_into_mse_and_std_reg():
    m = M.build_model(CFG, side=10, num_classes=18)
    fn = M.MDDCCLoss(CFG, 18)
    parts = fn(m(torch.rand(8, 4, 10, 10)), torch.randint(0, 18, (8,)), m)
    assert float(parts.total) == pytest.approx(
        float(parts.mse) + 1.0 * float(parts.std_reg), rel=1e-5)
    assert float(parts.std_reg) > 0


def test_lambda_zero_disables_regularizer():
    cfg = {**CFG, "loss": {**CFG["loss"],
                           "std_regularizer": {"enabled": False, "lambda_std": 1.0}}}
    m = M.build_model(CFG, side=10, num_classes=18)
    parts = M.MDDCCLoss(cfg, 18)(m(torch.rand(4, 4, 10, 10)),
                                 torch.randint(0, 18, (4,)), m)
    assert float(parts.std_reg) == 0.0
    assert float(parts.total) == pytest.approx(float(parts.mse), rel=1e-6)


def test_cross_entropy_is_rejected_for_main_run():
    cfg = {**CFG, "loss": {**CFG["loss"], "name": "cross_entropy"}}
    with pytest.raises(ValueError, match="mse"):
        M.MDDCCLoss(cfg, 18)


# ------------------------------------------------------------- optimizer
def test_optimizer_is_plain_sgd_without_weight_decay():
    m = M.build_model(CFG, side=10, num_classes=18)
    opt = M.build_optimizer(CFG, m)
    g = opt.param_groups[0]
    assert isinstance(opt, torch.optim.SGD)
    assert g["lr"] == 0.001 and g["momentum"] == 0.0
    assert g["weight_decay"] == 0.0, "muc 2.C: da co sigma(w), khong chong them L2"
    assert g["nesterov"] is False


def test_adam_is_rejected_for_main_run():
    cfg = {**CFG, "optim": {**CFG["optim"], "name": "adam"}}
    m = M.build_model(CFG, side=10, num_classes=18)
    with pytest.raises(ValueError, match="SGD"):
        M.build_optimizer(cfg, m)


def test_model_actually_learns_on_a_trivial_task():
    """Loss phai giam khi lap lai tren mot lo co dinh - bang chung BP chay dung."""
    torch.manual_seed(0)
    m = M.build_model(CFG, side=10, num_classes=3)
    cfg = {**CFG, "optim": {**CFG["optim"], "learning_rate": 0.5},
           "loss": {**CFG["loss"],
                    "std_regularizer": {"enabled": False, "lambda_std": 0.0}}}
    opt = M.build_optimizer(cfg, m)
    fn = M.MDDCCLoss(cfg, 3)

    x = torch.rand(64, 4, 10, 10)
    y = torch.randint(0, 3, (64,))
    first = last = None
    for _ in range(40):
        opt.zero_grad()
        parts = fn(m(x), y, m)
        parts.total.backward()
        opt.step()
        last = float(parts.mse)
        first = first if first is not None else last
    assert last < first, f"MSE khong giam: {first:.5f} -> {last:.5f}"


def test_grad_norm_is_positive_after_backward():
    m = M.build_model(CFG, side=10, num_classes=18)
    fn = M.MDDCCLoss(CFG, 18)
    fn(m(torch.rand(8, 4, 10, 10)), torch.randint(0, 18, (8,)), m).total.backward()
    assert M.grad_norm(m) > 0

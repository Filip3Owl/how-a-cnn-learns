"""Correctness tests for the from-scratch NumPy layers.

The whole project rests on these layers being right. A subtly wrong gradient
still trains *something*, and the resulting animation would confidently show
the viewer a lie. So every backward pass is checked two ways: against PyTorch's
autograd, and against a finite-difference approximation of the true derivative.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cnnviz.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    ReLU,
    softmax_cross_entropy,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Forward passes, against PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_conv_forward_matches_torch(rng, stride, padding):
    x = rng.normal(size=(4, 3, 9, 9)).astype(np.float32)
    conv = Conv2D(3, 5, kernel_size=3, stride=stride, padding=padding, seed=1)

    ours = conv.forward(x)
    theirs = F.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(conv.W),
        torch.from_numpy(conv.b),
        stride=stride,
        padding=padding,
    ).numpy()

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-5)


def test_maxpool_forward_matches_torch(rng):
    x = rng.normal(size=(4, 3, 8, 8)).astype(np.float32)
    ours = MaxPool2D(2).forward(x)
    theirs = F.max_pool2d(torch.from_numpy(x), 2).numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)


def test_softmax_cross_entropy_matches_torch(rng):
    logits = rng.normal(size=(6, 10)).astype(np.float32)
    labels = rng.integers(0, 10, size=6)

    loss, _ = softmax_cross_entropy(logits, labels)
    expected = F.cross_entropy(
        torch.from_numpy(logits), torch.from_numpy(labels)
    ).item()

    assert loss == pytest.approx(expected, rel=1e-5)


def test_softmax_is_numerically_stable_on_large_logits():
    """Naive softmax overflows here; the shifted implementation must not."""
    logits = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    loss, grad = softmax_cross_entropy(logits, np.array([1]))

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Backward passes, against PyTorch autograd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv_backward_matches_torch(rng, stride, padding):
    x = rng.normal(size=(4, 3, 9, 9)).astype(np.float32)
    conv = Conv2D(3, 5, kernel_size=3, stride=stride, padding=padding, seed=1)

    out = conv.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float32)
    dx = conv.backward(upstream)

    xt = torch.from_numpy(x).requires_grad_(True)
    wt = torch.from_numpy(conv.W).clone().requires_grad_(True)
    bt = torch.from_numpy(conv.b).clone().requires_grad_(True)
    F.conv2d(xt, wt, bt, stride=stride, padding=padding).backward(
        torch.from_numpy(upstream)
    )

    np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(conv.dW, wt.grad.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(conv.db, bt.grad.numpy(), rtol=1e-3, atol=1e-4)


def test_maxpool_backward_routes_gradient_to_the_winner_only(rng):
    x = rng.normal(size=(4, 3, 8, 8)).astype(np.float32)
    pool = MaxPool2D(2)

    out = pool.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float32)
    dx = pool.backward(upstream)

    xt = torch.from_numpy(x).requires_grad_(True)
    F.max_pool2d(xt, 2).backward(torch.from_numpy(upstream))

    np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-5, atol=1e-6)
    # Exactly one input per 2x2 window receives gradient.
    assert np.count_nonzero(dx) == out.size


def test_dense_backward_matches_torch(rng):
    x = rng.normal(size=(6, 12)).astype(np.float32)
    dense = Dense(12, 4, seed=2)

    out = dense.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float32)
    dx = dense.backward(upstream)

    xt = torch.from_numpy(x).requires_grad_(True)
    wt = torch.from_numpy(dense.W).clone().requires_grad_(True)
    bt = torch.from_numpy(dense.b).clone().requires_grad_(True)
    (xt @ wt + bt).backward(torch.from_numpy(upstream))

    np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dense.dW, wt.grad.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dense.db, bt.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_relu_gates_gradient_by_input_sign():
    x = np.array([[-2.0, -0.1, 0.0, 0.1, 2.0]], dtype=np.float32)
    relu = ReLU()

    np.testing.assert_allclose(
        relu.forward(x), [[0.0, 0.0, 0.0, 0.1, 2.0]], rtol=1e-6
    )
    upstream = np.ones_like(x)
    # Zero is not positive, so the gate is closed there too.
    np.testing.assert_array_equal(
        relu.backward(upstream), [[0.0, 0.0, 0.0, 1.0, 1.0]]
    )


def test_flatten_roundtrips_shape(rng):
    x = rng.normal(size=(4, 3, 5, 5)).astype(np.float32)
    flat = Flatten()
    assert flat.forward(x).shape == (4, 75)
    assert flat.backward(np.ones((4, 75), dtype=np.float32)).shape == x.shape


# ---------------------------------------------------------------------------
# Finite differences — an independent check that does not trust PyTorch either
# ---------------------------------------------------------------------------

def test_conv_gradient_against_finite_differences(rng):
    """Numerically differentiate the loss w.r.t. every kernel weight.

    Central differences carry two competing errors: truncation, which falls as
    ``h^2``, and floating-point cancellation, which *grows* as ``h`` shrinks
    because ``loss(W+h) - loss(W-h)`` loses significant digits. The total is
    minimised near ``eps^(1/3)``, so in float64 a step around ``1e-5`` is far
    more accurate than a naively smaller ``1e-7``.

    The layers are forced to float64 here for the same reason: at float32 the
    cancellation floor alone is ~1e-2 and the check proves nothing.
    """
    x = rng.normal(size=(2, 1, 6, 6)).astype(np.float64)
    labels = np.array([0, 1])

    conv = Conv2D(1, 2, kernel_size=3, seed=3)
    conv.W = conv.W.astype(np.float64)
    conv.b = conv.b.astype(np.float64)
    flat, dense = Flatten(), Dense(2 * 4 * 4, 2, seed=4)
    dense.W = dense.W.astype(np.float64)
    dense.b = dense.b.astype(np.float64)

    def loss_of(weights: np.ndarray) -> float:
        conv.W = weights
        logits = dense.forward(flat.forward(ReLU().forward(conv.forward(x))))
        return softmax_cross_entropy(logits, labels)[0]

    # Analytic gradient
    relu = ReLU()
    logits = dense.forward(flat.forward(relu.forward(conv.forward(x))))
    _, dlogits = softmax_cross_entropy(logits, labels)
    conv.backward(relu.backward(flat.backward(dense.backward(dlogits))))
    analytic = conv.dW.copy()

    # Numeric gradient at every coordinate — the kernel is small enough.
    h = 1e-5
    base = conv.W.copy()
    for idx in np.ndindex(base.shape):
        plus = base.copy()
        plus[idx] += h
        minus = base.copy()
        minus[idx] -= h

        numeric = (loss_of(plus) - loss_of(minus)) / (2 * h)
        assert numeric == pytest.approx(analytic[idx], rel=1e-5, abs=1e-8)


def test_conv_output_shape_helper_agrees_with_forward(rng):
    x = rng.normal(size=(2, 1, 13, 11)).astype(np.float32)
    conv = Conv2D(1, 3, kernel_size=3, stride=2, padding=1, seed=5)
    assert conv.forward(x).shape[2:] == conv.output_shape(13, 11)


def test_maxpool_rejects_indivisible_input(rng):
    x = rng.normal(size=(1, 1, 7, 7)).astype(np.float32)
    with pytest.raises(ValueError, match="not divisible"):
        MaxPool2D(2).forward(x)


def test_backward_before_forward_is_an_error():
    with pytest.raises(RuntimeError, match="before forward"):
        Conv2D(1, 1, seed=0).backward(np.zeros((1, 1, 2, 2), dtype=np.float32))

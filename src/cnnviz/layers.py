"""Convolutional network layers written from scratch in NumPy.

These exist to be *read*, not to be fast. Every forward and backward pass is
written out explicitly so the notebooks can open any layer mid-computation and
animate exactly what it holds. Where a vectorised trick would obscure the
mathematics, the loop is kept.

Each layer follows the same contract:

* ``forward(x)`` returns the output and stashes whatever the backward pass
  needs on ``self.cache``.
* ``backward(grad_output)`` returns the gradient with respect to the input and
  stores parameter gradients on the layer.

That symmetry is the point: backpropagation is the same graph as the forward
pass, walked in reverse.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "Conv2D",
    "ReLU",
    "MaxPool2D",
    "Flatten",
    "Dense",
    "Sequential",
    "softmax_cross_entropy",
    "receptive_field",
    "trace_receptive_field",
]


class Layer:
    """Base class defining the forward/backward contract."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Trainable tensors, keyed by name. Empty for stateless layers."""
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        """Gradients matching :attr:`params`, populated by ``backward``."""
        return {}


class Conv2D(Layer):
    r"""2-D cross-correlation with learnable kernels and biases.

    For output channel :math:`m`, the forward pass computes

    .. math::

        y_{m,i,j} = b_m + \sum_{c=0}^{C-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1}
                    w_{m,c,u,v} \; x_{c,\, is+u-p,\, js+v-p}

    Note this is *cross-correlation*, not true convolution — the kernel is not
    flipped. Every deep-learning framework does the same; since the kernel is
    learned, the flip is absorbed into the learned values and only the
    terminology suffers.

    Args:
        in_channels: Channels of the input.
        out_channels: Number of kernels, i.e. channels of the output.
        kernel_size: Side length of the square kernel.
        stride: Step between successive kernel placements.
        padding: Zero-padding added to each spatial edge.
        seed: Seed for reproducible He initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/fan_in keeps activation scale stable
        # through ReLU layers, which zero out half the distribution.
        fan_in = in_channels * kernel_size * kernel_size
        self.W = rng.normal(
            0.0, np.sqrt(2.0 / fan_in),
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache: np.ndarray | None = None

    @staticmethod
    def _pad(x: np.ndarray, p: int) -> np.ndarray:
        if p == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    def output_shape(self, height: int, width: int) -> tuple[int, int]:
        """Spatial size of the output for a given input size."""
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out_h, out_w

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Slide every kernel across the input.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            Output of shape ``(N, M, H_out, W_out)``.
        """
        self.cache = x
        n, _, h, w = x.shape
        m, _, k, _ = self.W.shape
        out_h, out_w = self.output_shape(h, w)

        x_padded = self._pad(x, self.padding)
        # Derive the dtype rather than hardcoding float32: promoting to the
        # input's precision is what lets the float64 gradient check in the
        # tests reach finite-difference accuracy instead of stalling at ~1e-7.
        out = np.zeros((n, m, out_h, out_w), dtype=np.result_type(x, self.W))

        # Loop over output positions, not over the batch: at each position we
        # take one patch across all images and all channels at once. This is
        # the shape of the computation the animations depict.
        for i in range(out_h):
            for j in range(out_w):
                top, left = i * self.stride, j * self.stride
                patch = x_padded[:, :, top:top + k, left:left + k]
                # (N,1,C,k,k) * (1,M,C,k,k) summed over C,k,k -> (N,M)
                out[:, :, i, j] = np.tensordot(
                    patch, self.W, axes=([1, 2, 3], [1, 2, 3])
                )

        return out + self.b[None, :, None, None]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Distribute the incoming gradient to kernels, biases and input.

        Three gradients come out of one loop:

        * ``db`` — sum of the incoming gradient over batch and space, since
          the bias is added identically at every position.
        * ``dW`` — correlate the incoming gradient against the input patches:
          each kernel weight accumulates from every position it touched.
        * ``dx`` — scatter each weight's contribution back to the pixels it
          read. Overlapping receptive fields *accumulate*, which is why this
          is ``+=`` and not ``=``.
        """
        if self.cache is None:
            raise RuntimeError("backward() called before forward().")

        x = self.cache
        n, c, h, w = x.shape
        m, _, k, _ = self.W.shape
        _, _, out_h, out_w = grad_output.shape

        x_padded = self._pad(x, self.padding)
        dx_padded = np.zeros_like(x_padded)
        self.dW = np.zeros_like(self.W)
        self.db = grad_output.sum(axis=(0, 2, 3))

        for i in range(out_h):
            for j in range(out_w):
                top, left = i * self.stride, j * self.stride
                patch = x_padded[:, :, top:top + k, left:left + k]  # (N,C,k,k)
                g = grad_output[:, :, i, j]                          # (N,M)

                self.dW += np.tensordot(g, patch, axes=([0], [0]))
                dx_padded[:, :, top:top + k, left:left + k] += np.tensordot(
                    g, self.W, axes=([1], [0])
                )

        if self.padding:
            p = self.padding
            return dx_padded[:, :, p:-p, p:-p]
        return dx_padded

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}


class ReLU(Layer):
    r"""Rectified linear unit, :math:`f(x) = \max(0, x)`.

    The backward pass is a gate: gradient flows through wherever the input was
    positive and is blocked everywhere else. Animating this mask is the
    clearest way to show a unit "dying" — once its input is negative for every
    example, no gradient reaches it again.
    """

    def __init__(self) -> None:
        self.cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() called before forward().")
        return grad_output * self.cache


class MaxPool2D(Layer):
    """Downsample by taking the maximum of each non-overlapping window.

    The backward pass routes the whole gradient of a window to the single
    input that won the max; every other input in the window receives zero.
    The winner's index is recorded during the forward pass rather than
    recomputed, which is both faster and what frameworks actually do.
    """

    def __init__(self, size: int = 2) -> None:
        self.size = size
        self.cache: tuple[np.ndarray, tuple[int, ...]] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        s = self.size
        if h % s or w % s:
            raise ValueError(
                f"Input {h}x{w} is not divisible by pool size {s}. "
                "Pad the input or change the pool size."
            )
        out_h, out_w = h // s, w // s

        # Reshape into windows, then reduce the two window axes.
        windows = x.reshape(n, c, out_h, s, out_w, s).transpose(0, 1, 2, 4, 3, 5)
        flat = windows.reshape(n, c, out_h, out_w, s * s)
        argmax = flat.argmax(axis=-1)

        self.cache = (argmax, x.shape)
        return flat.max(axis=-1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() called before forward().")

        argmax, shape = self.cache
        n, c, h, w = shape
        s = self.size
        out_h, out_w = h // s, w // s

        flat = np.zeros((n, c, out_h, out_w, s * s), dtype=grad_output.dtype)
        np.put_along_axis(flat, argmax[..., None], grad_output[..., None], axis=-1)

        windows = flat.reshape(n, c, out_h, out_w, s, s).transpose(0, 1, 2, 4, 3, 5)
        return windows.reshape(n, c, h, w)


class Flatten(Layer):
    """Collapse ``(N, C, H, W)`` to ``(N, C*H*W)``; backward restores shape."""

    def __init__(self) -> None:
        self.cache: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() called before forward().")
        return grad_output.reshape(self.cache)


class Dense(Layer):
    r"""Fully connected layer, :math:`y = xW + b`.

    Gradients follow from the chain rule on a matrix product:

    .. math::

        \frac{\partial L}{\partial W} = x^{\top} \frac{\partial L}{\partial y},
        \qquad
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^{\top}
    """

    def __init__(
        self, in_features: int, out_features: int, seed: int | None = None
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(
            0.0, np.sqrt(2.0 / in_features), size=(in_features, out_features)
        ).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("backward() called before forward().")
        x = self.cache
        self.dW = x.T @ grad_output
        self.db = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}


class Sequential(Layer):
    """An ordered stack of layers that keeps every intermediate result.

    A framework's ``Sequential`` throws the intermediates away as soon as the
    forward pass is done. Here they are the product: :attr:`activations` holds
    the input followed by the output of each layer, which is exactly the list
    of panels a "what does depth do" animation draws.

    Args:
        *layers: Layers to apply in order.
    """

    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)
        #: Input first, then one entry per layer — populated by ``forward``.
        self.activations: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def trace(self) -> list[tuple[str, tuple[int, ...]]]:
        """``(label, shape)`` for the input and every layer of the last pass.

        The shape ladder a notebook prints beside the pipeline. Raises if no
        forward pass has run, rather than reporting shapes it cannot know.
        """
        if not self.activations:
            raise RuntimeError("trace() called before forward().")

        rows = [("input", self.activations[0].shape)]
        for layer, output in zip(self.layers, self.activations[1:], strict=True):
            rows.append((type(layer).__name__, output.shape))
        return rows

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Every trainable tensor, keyed ``"<index>.<name>"``.

        The arrays are the layers' own, not copies, so an optimiser can update
        them in place — ``params["0.W"] -= lr * grads["0.W"]`` trains the stack.
        """
        return {
            f"{i}.{name}": value
            for i, layer in enumerate(self.layers)
            for name, value in layer.params.items()
        }

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{name}": value
            for i, layer in enumerate(self.layers)
            for name, value in layer.grads.items()
        }


# --------------------------------------------------------------------------
# Receptive fields
#
# The receptive field of a unit is the patch of the *input* that can still
# change its value. It is the quantitative form of "depth buys context": a
# layer-1 unit sees a few pixels, a layer-4 unit sees most of the digit, and
# nothing but the arithmetic of kernel size, stride and padding decides that.
# --------------------------------------------------------------------------

def _geometry(layer: Layer) -> tuple[int, int, int] | None:
    """``(kernel, stride, padding)`` for a layer, or None if it moves no window.

    ReLU and other elementwise layers return None: they read one input per
    output and so leave the geometry untouched.
    """
    if isinstance(layer, Conv2D):
        return layer.kernel_size, layer.stride, layer.padding
    if isinstance(layer, MaxPool2D):
        return layer.size, layer.size, 0
    if isinstance(layer, (Flatten, Dense)):
        raise ValueError(
            f"{type(layer).__name__} destroys the spatial grid; trace the "
            "receptive field over the convolutional part of the stack only."
        )
    return None


def _geometries(stack: Iterable[Layer], depth: int | None = None):
    layers = list(stack)
    if depth is not None:
        layers = layers[:depth]
    return [g for g in map(_geometry, layers) if g is not None]


def _span(geometries: Sequence[tuple[int, int, int]], index: int) -> tuple[int, int]:
    """First and last input index read by output ``index``, walking backwards.

    One layer at a time, output ``i`` reads inputs ``i*s - p`` through
    ``i*s - p + k - 1``. Composing that from the deepest layer outwards is
    exact, including the negative indices that mean "this unit is partly
    looking at padding rather than at the image".
    """
    low = high = index
    for kernel, stride, padding in reversed(geometries):
        low = low * stride - padding
        high = high * stride + kernel - 1 - padding
    return low, high


def receptive_field(stack: Iterable[Layer], depth: int | None = None) -> tuple[int, int]:
    """Size and spacing, in input pixels, of one unit's receptive field.

    Args:
        stack: The layers, in forward order. A :class:`Sequential` works
            directly, as does any iterable of layers.
        depth: Consider only the first ``depth`` layers. ``None`` uses all
            of them.

    Returns:
        ``(size, jump)`` — the side length of the input patch a unit sees, and
        how many input pixels apart neighbouring units' patches sit. ``jump``
        is the product of the strides, and is what makes a deep unit's view
        grow faster than its kernels: after two 2x2 pools, a 5x5 kernel reaches
        across 20 input pixels, not 5.
    """
    geometries = _geometries(stack, depth)
    low0, high0 = _span(geometries, 0)
    low1, _ = _span(geometries, 1)
    return high0 - low0 + 1, low1 - low0


def trace_receptive_field(
    stack: Iterable[Layer],
    row: int,
    col: int,
    depth: int | None = None,
) -> tuple[int, int, int]:
    """Locate, in input pixels, the patch feeding one output unit.

    Args:
        stack: The layers, in forward order.
        row: Row of the unit in the output of layer ``depth``.
        col: Its column.
        depth: Consider only the first ``depth`` layers.

    Returns:
        ``(top, left, size)`` — ready to hand to
        :func:`cnnviz.panels.highlight_window`. ``top`` or ``left`` can be
        negative when the unit reads into the zero padding rather than the
        image; clip before indexing, but draw the box as returned, since the
        padding is genuinely part of what that unit reads.
    """
    geometries = _geometries(stack, depth)
    top, bottom = _span(geometries, row)
    left, right = _span(geometries, col)

    size = bottom - top + 1
    if right - left + 1 != size:  # pragma: no cover - square kernels only
        raise ValueError("Non-square receptive field; this helper assumes square.")
    return top, left, size


def softmax_cross_entropy(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    r"""Softmax followed by cross-entropy loss, fused.

    Fusing them matters. Computed separately, the gradient involves a division
    by a probability that can underflow to zero. Together, the gradient
    collapses to the famously simple

    .. math::

        \frac{\partial L}{\partial z_i} = \frac{p_i - y_i}{N}

    — the predicted distribution minus the true one. That difference *is* the
    error signal every animation in this project traces backwards.

    Args:
        logits: Raw scores of shape ``(N, K)``.
        labels: Integer class indices of shape ``(N,)``.

    Returns:
        The mean loss, and the gradient with respect to ``logits``.
    """
    # Subtract the row max before exponentiating: mathematically a no-op,
    # numerically the difference between a result and an overflow to inf.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]
    loss = float(-np.log(probs[np.arange(n), labels] + 1e-12).mean())

    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n

    return loss, grad

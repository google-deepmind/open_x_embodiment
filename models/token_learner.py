"""Token Learner implementation from https://arxiv.org/abs/2106.11297.

Implementation from github.com/google-research/scenic/projects/token_learner/.
"""

from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(inputs)
    x = IdentityLayer(name='mlp1')(self.activation_fn(x))
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(x)
    output = IdentityLayer(name='mlp2')(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic
    )
    return output


class TokenLearnerModuleV11(nn.Module):
  """TokenLearner module Version 1.1, using slightly different conv. layers.

  Instead of using 4 conv. layers with small channels to implement spatial
  attention, this version uses a MLP with gelu inbetween. It also uses softmax
  instead of sigmoid. We confirmed that this version works better in general.

  Attributes:
    num_tokens: Number of tokens.
    bottleneck_dim: The size of hidden units in the MLP for spatial attention.
    dropout_rate: Dropout rate.
  """

  num_tokens: int
  bottleneck_dim: int = 64
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies learnable tokenization to the 2D inputs.

    Args:
      inputs: Inputs of shape `[bs, h, w, c]`.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Output of shape `[bs, n_token, c]`.
    """
    if inputs.ndim == 4:
      n, h, w, c = inputs.shape
      inputs = jnp.reshape(inputs, [n, h * w, c])

    feature_shape = inputs.shape

    selected = inputs

    selected = nn.LayerNorm()(selected)

    selected = MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=self.num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_masking',
    )(selected, deterministic=deterministic)

    selected = jnp.reshape(
        selected, [feature_shape[0], -1, self.num_tokens]
    )  # Shape: [bs, h*w, n_token].
    selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
    selected = jax.nn.softmax(selected, axis=-1)

    feat = inputs
    feat = jnp.reshape(
        feat, [feature_shape[0], -1, feature_shape[-1]]
    )  # Shape: [bs, h*w, c].

    feat = jnp.einsum('...si,...id->...sd', selected, feat)

    return feat

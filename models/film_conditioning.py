"""Film-Conditioning layer.

Related papers:
- https://arxiv.org/abs/1709.07871
"""

import flax.linen as nn


class FilmConditioning(nn.Module):
  """FiLM conditioning layer."""

  num_channels: int

  @nn.compact
  def __call__(self, conv_filters, context):
    """Applies FiLM conditioning to the input.

    Args:
      conv_filters: array of shape (B, H, W, C), usually an output conv feature
        map.
      context: array of shape (B, context_size).

    Returns:
      array of shape (B, H, W, C) with the FiLM conditioning applied.
    """
    zero_init = nn.initializers.zeros_init()
    project_cond_add = nn.Dense(
        self.num_channels, kernel_init=zero_init, bias_init=zero_init
    )(context)
    project_cond_mul = nn.Dense(
        self.num_channels, kernel_init=zero_init, bias_init=zero_init
    )(context)

    project_cond_add = project_cond_add[:, None, None, :]
    project_cond_mul = project_cond_mul[:, None, None, :]

    result = (1 + project_cond_mul) * conv_filters + project_cond_add
    return result

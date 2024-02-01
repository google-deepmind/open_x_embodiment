"""Defines variants of the EfficientNet model.

Currently implements the follow variants:
- EfficientNetWithFilm - EfficientNet backbone with FiLM-conditioning applied.
"""

import copy
import math
from typing import Any, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import initializers
import jax
from jax import numpy as jnp

from . import film_conditioning


MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


# Relevant initializers. The original implementation uses fan_out Kaiming init.

conv_kernel_init_fn = initializers.variance_scaling(2.0, 'fan_out', 'normal')

dense_kernel_init_fn = initializers.variance_scaling(
    1 / 3.0, 'fan_out', 'uniform'
)


class DepthwiseConv(nn.Module):
  """Depthwise convolution that matches tensorflow's conventions.

  In Tensorflow, the shapes of depthwise kernels don't match the shapes of a
  regular convolutional kernel of appropriate feature_group_count.
  It is safer to use this class instead of the regular Conv (easier port of
  tensorflow checkpoints, fan_out initialization of the previous layer will
  match the tensorflow behavior, etc...).

  Attributes:
    features: Number of convolution filters.
    kernel_size: Shape of the convolutional kernel.
    strides: A sequence of `n` integers, representing the inter-window strides.
    padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of `inputs`. Convolution with
      input dilation `d` is equivalent to transposed convolution with stride
      `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of the convolution kernel.
      Convolution with kernel dilation is also known as 'atrous convolution'.
    feature_group_count: Unused attribute present in nn.Conv. Declare it to
      match the nn.Conv API.
    use_bias: Whether to add a bias to the output (default: True).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: Initializer for the convolutional kernel.
    bias_init: Initializer for the bias.
  """

  features: int
  kernel_size: Tuple[int, int]
  strides: Optional[Tuple[int, int]] = None
  padding: Union[str, Sequence[int]] = 'SAME'
  input_dilation: Optional[Sequence[int]] = None
  kernel_dilation: Optional[Sequence[int]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: Any = None
  kernel_init: Any = initializers.lecun_normal()
  bias_init: Any = initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a convolution to the inputs.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).

    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    in_features = inputs.shape[-1]
    strides = self.strides

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    kernel_shape = self.kernel_size + (self.features, 1)
    # Naming convention follows tensorflow.
    kernel = self.param('depthwise_kernel', self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, self.dtype)

    # Need to transpose to convert tensorflow-shaped kernel to lax-shaped kernel
    kernel = jnp.transpose(kernel, [0, 1, 3, 2])

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint:disable=protected-access

    y = jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        self.padding,
        lhs_dilation=self.input_dilation,
        rhs_dilation=self.kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=in_features,
        precision=self.precision,
    )

    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias

    return y


# pytype: disable=attribute-error
# pylint:disable=unused-argument
class BlockConfig(object):
  """Class that contains configuration parameters for a single block."""

  def __init__(
      self,
      input_filters: int = 0,
      output_filters: int = 0,
      kernel_size: int = 3,
      num_repeat: int = 1,
      expand_ratio: int = 1,
      strides: Tuple[int, int] = (1, 1),
      se_ratio: Optional[float] = None,
      id_skip: bool = True,
      fused_conv: bool = False,
      conv_type: str = 'depthwise',
  ):
    for arg in locals().items():
      setattr(self, *arg)


class ModelConfig(object):
  """Class that contains configuration parameters for the model."""

  def __init__(
      self,
      width_coefficient: float = 1.0,
      depth_coefficient: float = 1.0,
      resolution: int = 224,
      dropout_rate: float = 0.2,
      blocks: Tuple[BlockConfig, ...] = (
          # (input_filters, output_filters, kernel_size, num_repeat,
          #  expand_ratio, strides, se_ratio)
          # pylint: disable=bad-whitespace
          BlockConfig(32, 16, 3, 1, 1, (1, 1), 0.25),
          BlockConfig(16, 24, 3, 2, 6, (2, 2), 0.25),
          BlockConfig(24, 40, 5, 2, 6, (2, 2), 0.25),
          BlockConfig(40, 80, 3, 3, 6, (2, 2), 0.25),
          BlockConfig(80, 112, 5, 3, 6, (1, 1), 0.25),
          BlockConfig(112, 192, 5, 4, 6, (2, 2), 0.25),
          BlockConfig(192, 320, 3, 1, 6, (1, 1), 0.25),
          # pylint: enable=bad-whitespace
      ),
      stem_base_filters: int = 32,
      top_base_filters: int = 1280,
      activation: str = 'swish',
      batch_norm: str = 'default',
      bn_momentum: float = 0.99,
      bn_epsilon: float = 1e-3,
      # While the original implementation used a weight decay of 1e-5,
      # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
      weight_decay: float = 5e-6,
      drop_connect_rate: float = 0.2,
      depth_divisor: int = 8,
      min_depth: Optional[int] = None,
      use_se: bool = True,
      input_channels: int = 3,
      num_classes: int = 1000,
      model_name: str = 'efficientnet',
      rescale_input: bool = True,
      data_format: str = 'channels_last',
      final_projection_size: int = 0,
      classifier_head: bool = True,
      dtype: jnp.dtype = jnp.float32,
  ):
    """Default Config for Efficientnet-B0."""
    for arg in locals().items():
      setattr(self, *arg)


# pylint:enable=unused-argument


MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'efficientnet-b3': ModelConfig(1.2, 1.4, 300, 0.3),
}


def round_filters(filters: int, config: ModelConfig) -> int:
  """Returns rounded number of filters based on width coefficient."""
  width_coefficient = config.width_coefficient
  min_depth = config.min_depth
  divisor = config.depth_divisor

  if not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
  """Returns rounded number of repeats based on depth coefficient."""
  return int(math.ceil(depth_coefficient * repeats))


def conv2d(
    inputs: jnp.ndarray,
    num_filters: int,
    config: ModelConfig,
    kernel_size: Union[int, Tuple[int, int]] = (1, 1),
    strides: Tuple[int, int] = (1, 1),
    use_batch_norm: bool = True,
    use_bias: bool = False,
    activation: Any = None,
    depthwise: bool = False,
    train: bool = True,
    conv_name: Optional[str] = None,
    bn_name: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
  """Convolutional layer with possibly batch norm and activation.

  Args:
    inputs: Input data with dimensions (batch, spatial_dims..., features).
    num_filters: Number of convolution filters.
    config: Configuration for the model.
    kernel_size: Size of the kernel, as a tuple of int.
    strides: Strides for the convolution, as a tuple of int.
    use_batch_norm: Whether batch norm should be applied to the output.
    use_bias: Whether we should add bias to the output of the first convolution.
    activation: Name of the activation function to use.
    depthwise: If true, will use depthwise convolutions.
    train: Whether the model should behave in training or inference mode.
    conv_name: Name to give to the convolution layer.
    bn_name: Name to give to the batch norm layer.
    dtype: dtype for the computation.

  Returns:
    The output of the convolutional layer.
  """
  conv_fn = DepthwiseConv if depthwise else nn.Conv
  kernel_size = (
      (kernel_size, kernel_size)
      if isinstance(kernel_size, int)
      else tuple(kernel_size)
  )
  conv_name = conv_name if conv_name else 'conv2d'
  bn_name = bn_name if bn_name else 'batch_normalization'

  x = conv_fn(
      num_filters,
      kernel_size,
      tuple(strides),
      padding='SAME',
      use_bias=use_bias,
      kernel_init=conv_kernel_init_fn,
      name=conv_name,
      dtype=dtype,
  )(inputs)

  if use_batch_norm:
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=config.bn_momentum,
        epsilon=config.bn_epsilon,
        name=bn_name,
        dtype=dtype,
    )(x)

  if activation is not None:
    x = getattr(nn.activation, activation.lower())(x)
  return x


def stochastic_depth(
    inputs: jnp.ndarray,
    rng: jnp.ndarray,
    survival_probability: float,
    deterministic: bool = False,
) -> jnp.ndarray:
  """Applies stochastic depth.

  Args:
    inputs: The inputs that should be randomly masked.
    rng: A `jax.random.PRNGKey`.
    survival_probability: 1 - the probability of masking out a value.
    deterministic: If false the inputs are scaled by `1 / (1 - rate)` and
      masked, whereas if true, no mask is applied and the inputs are returned as
      is.

  Returns:
    The masked inputs.
  """
  if survival_probability == 1.0 or deterministic:
    return inputs

  mask_shape = [inputs.shape[0]] + [1 for _ in inputs.shape[1:]]
  mask = jax.random.bernoulli(rng, p=survival_probability, shape=mask_shape)
  mask = jnp.tile(mask, [1] + list(inputs.shape[1:]))
  return jax.lax.select(
      mask, inputs / survival_probability, jnp.zeros_like(inputs)
  )


class SqueezeExcite(nn.Module):
  """SqueezeExite block (See: https://arxiv.org/abs/1709.01507.)

  Attributes:
    num_filters: Number of convolution filters.
    block: Configuration for this block.
    config: Configuration for the model.
    train: Whether the model is in training or inference mode.
  """

  num_filters: int
  block: BlockConfig
  config: ModelConfig
  train: bool

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a convolution to the inputs.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).

    Returns:
      The output of the squeeze excite block.
    """
    block = self.block
    config = self.config
    train = self.train
    dtype = config.dtype
    num_reduced_filters = max(1, int(block.input_filters * block.se_ratio))

    se = nn.avg_pool(inputs, inputs.shape[1:3])
    se = conv2d(
        se,
        num_reduced_filters,
        config,
        use_bias=True,
        use_batch_norm=False,
        activation=config.activation,
        conv_name='reduce_conv2d_0',
        train=train,
        dtype=dtype,
    )

    se = conv2d(
        se,
        self.num_filters,
        config,
        use_bias=True,
        use_batch_norm=False,
        activation='sigmoid',
        conv_name='expand_conv2d_0',
        train=train,
        dtype=dtype,
    )

    return inputs * se


class MBConvBlock(nn.Module):
  """Main building component of Efficientnet.

  Attributes:
    block: BlockConfig, arguments to create a Block.
    config: ModelConfig, a set of model parameters.
    train: Whether we are training or predicting.
  """

  block: BlockConfig
  config: ModelConfig
  train: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Mobile Inverted Residual Bottleneck.

    Args:
      inputs: Input to the block.

    Returns:
      The output of the block.
    """
    config = self.config
    block = self.block
    train = self.train
    use_se = config.use_se
    activation = config.activation
    drop_connect_rate = config.drop_connect_rate
    use_depthwise = block.conv_type != 'no_depthwise'
    dtype = config.dtype

    rng = self.make_rng('random')

    filters = block.input_filters * block.expand_ratio

    x = inputs
    bn_index = 0

    if block.fused_conv:
      # If we use fused mbconv, skip expansion and use regular conv.
      x = conv2d(
          x,
          filters,
          config,
          kernel_size=block.kernel_size,
          strides=block.strides,
          activation=activation,
          conv_name='fused_conv2d_0',
          bn_name='batch_normalization_' + str(bn_index),
          train=train,
          dtype=dtype,
      )
      bn_index += 1
    else:
      if block.expand_ratio != 1:
        # Expansion phase
        kernel_size = (1, 1) if use_depthwise else (3, 3)
        x = conv2d(
            x,
            filters,
            config,
            kernel_size=kernel_size,
            activation=activation,
            conv_name='expand_conv2d_0',
            bn_name='batch_normalization_' + str(bn_index),
            train=train,
            dtype=dtype,
        )
        bn_index += 1
      # Depthwise Convolution
      if use_depthwise:
        x = conv2d(
            x,
            num_filters=x.shape[-1],  # Depthwise conv
            config=config,
            kernel_size=block.kernel_size,
            strides=block.strides,
            activation=activation,
            depthwise=True,
            conv_name='depthwise_conv2d',
            bn_name='batch_normalization_' + str(bn_index),
            train=train,
            dtype=dtype,
        )
        bn_index += 1

    # Squeeze and Excitation phase
    if use_se:
      assert block.se_ratio is not None
      assert 0 < block.se_ratio <= 1
      x = SqueezeExcite(
          num_filters=filters, block=block, config=config, train=train
      )(x)

    # Output phase
    x = conv2d(
        x,
        block.output_filters,
        config,
        activation=None,
        conv_name='project_conv2d_0',
        bn_name='batch_normalization_' + str(bn_index),
        train=train,
        dtype=dtype,
    )

    if (
        block.id_skip
        and all(s == 1 for s in block.strides)
        and block.input_filters == block.output_filters
    ):
      if drop_connect_rate and drop_connect_rate > 0:
        survival_probability = 1 - drop_connect_rate
        x = stochastic_depth(
            x, rng, survival_probability, deterministic=not train
        )
      x = x + inputs

    return x


class Stem(nn.Module):
  """Initial block of Efficientnet.

  Attributes:
    config: ModelConfig, a set of model parameters.
    train: Whether we are training or predicting.
  """

  config: ModelConfig
  train: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Returns the output of the stem block.

    Args:
      inputs: The input to the block.

    Returns:
      Output of the block
    """
    config = self.config
    train = self.train
    x = conv2d(
        inputs,
        round_filters(config.stem_base_filters, config),
        config,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=config.activation,
        train=train,
        dtype=config.dtype,
    )
    return x


class Head(nn.Module):
  """Final block of Efficientnet.

  Attributes:
    config: A set of model parameters.
    train: Whether we are training or predicting.
  """

  config: Any
  train: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Returns the output of the head block.

    Args:
      inputs: The input to the block.

    Returns:
      x: Classifier logits.
    """
    config = self.config
    train = self.train
    dtype = config.dtype
    # Build top.
    x = conv2d(
        inputs,
        round_filters(config.top_base_filters, config),
        config,
        activation=config.activation,
        train=train,
        dtype=dtype,
    )
    return x
# pytype: enable=attribute-error


class EfficientNetWithFilm(nn.Module):
  """EfficientNet with FiLM conditioning."""

  config: Any
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, context_input: jnp.ndarray, *, train: bool
  ):
    """Returns the output of the EfficientNet model."""
    config = copy.deepcopy(self.config)
    config.dtype = self.dtype
    depth_coefficient = config.depth_coefficient
    blocks = config.blocks
    drop_connect_rate = config.drop_connect_rate

    inputs = jnp.asarray(inputs, self.dtype)

    # Build stem.
    x = Stem(config=config, train=train)(inputs)

    # Build blocks.
    num_blocks_total = sum(
        round_repeats(block.num_repeat, depth_coefficient) for block in blocks
    )
    block_num = 0

    for _, block in enumerate(blocks):
      assert block.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block.input_filters = round_filters(block.input_filters, config)
      block.output_filters = round_filters(block.output_filters, config)
      block.num_repeat = round_repeats(block.num_repeat, depth_coefficient)

      # The first block needs to take care of stride and filter size increase
      drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
      config.drop_connect_rate = drop_rate

      x = MBConvBlock(block=block, config=config, train=train)(x)

      x = film_conditioning.FilmConditioning(num_channels=x.shape[-1])(
          x, context_input
      )

      block_num += 1
      if block.num_repeat > 1:
        block.input_filters = block.output_filters
        block.strides = [1, 1]

        for _ in range(block.num_repeat - 1):
          drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
          config.drop_connect_rate = drop_rate
          x = MBConvBlock(block=block, config=config, train=train)(x)
          x = film_conditioning.FilmConditioning(num_channels=x.shape[-1])(
              x, context_input
          )

          block_num += 1

    x = Head(self.config, train=train)(x)

    return x

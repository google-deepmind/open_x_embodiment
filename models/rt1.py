"""Jax implementation of RT-1 / RT-1-X (https://arxiv.org/abs/2212.06817).

This is based on the tensorflow implementation in:
https://github.com/google-research/robotics_transformer and also includes
improvements made in RT-X (https://arxiv.org/abs/2310.08864).
"""

import enum
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from . import efficientnet
from . import film_conditioning
from . import token_learner


class FFNOptions(enum.Enum):
  """Different choices of FFN block for ablation testing."""

  LINEAR = 'linear'  # RT-1 Legacy
  SWIGLU = 'swiglu'  # Match LLaMa


class TransformerBlock(nn.Module):
  """A self-attention transformer block.

  See the `_TransformerLayer` in
  google-research/robotics_transformer/transformer.py for the original
  tensorflow implementation.
  """
  layer_size: int = 128
  num_heads: int = 8
  feed_forward_hidden_size: int = 512
  feed_forward_output_size: int = 512
  ffn_option: FFNOptions = FFNOptions.SWIGLU
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
    x1 = nn.LayerNorm()(x)

    x1 = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=(self.layer_size * self.num_heads),
        dropout_rate=self.dropout_rate,
    )(x1, x1, mask=attn_mask, deterministic=not train)

    x = x + x1

    y = nn.LayerNorm()(x)

    if self.ffn_option == FFNOptions.SWIGLU:
      h1 = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
      h1 = nn.swish(h1)
      gate = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
      ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(h1 * gate)
    elif self.ffn_option == FFNOptions.LINEAR:
      ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(y)
    else:
      raise ValueError(f'Unknown FFN option: {self.ffn_option}')

    ff_y = nn.Dropout(self.dropout_rate)(ff_y, deterministic=not train)
    x = x + ff_y
    return x


class Transformer(nn.Module):
  """Transformer architecture with dense positional embedding.

  See the `Transformer` in
  google-research/robotics_transformer/transformer.py for the original
  tensorflow implementation.
  """

  num_layers: int = 8
  layer_size: int = 128
  num_heads: int = 8
  feed_forward_hidden_size: int = 512
  feed_forward_output_size: int = 512
  ffn_option: FFNOptions = FFNOptions.SWIGLU
  dropout_rate: float = 0.1
  vocab_size: int = 256

  @nn.compact
  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
    bs, seqlen, *_ = x.shape

    pos = jnp.expand_dims(jnp.arange(0, seqlen, 1), 0)
    pos = jnp.tile(pos, [bs, 1])
    pos = jax.nn.one_hot(pos, seqlen)

    x = nn.Dense(self.feed_forward_output_size)(x)
    pos_emb = nn.Dense(self.feed_forward_output_size)(pos)
    x += pos_emb

    for _ in range(self.num_layers):
      x = TransformerBlock(
          layer_size=self.layer_size,
          num_heads=self.num_heads,
          feed_forward_hidden_size=self.feed_forward_hidden_size,
          feed_forward_output_size=self.feed_forward_output_size,
          dropout_rate=self.dropout_rate,
          ffn_option=self.ffn_option,
      )(x, attn_mask, train=train)

    output_tokens = nn.Dense(self.vocab_size)(x)
    return output_tokens


class ImageTokenizer(nn.Module):
  """Tokenizes images with EfficientNet+FiLM.

  This is based on the `RT1ImageTokenizer` implementation here:
  google-research/robotics_transformer/tokenizers/image_tokenizer.py

  The overall flow of the image tokenizer:
  * The input image batch dimensions are squashed, and the image is normalized.
  * The image is fed through the `EfficientNetWithFilm`.
  * A 1x1 convolution is applied to project to `num_features`.
  * Another final `FilmConditioning` layer is applied with the context.
  * `TokenLearnerModuleV11` is applied to project the tokens to `num_tokens`.
  """

  num_tokens: int = 8
  num_features: int = 512

  use_token_learner: bool = True

  @nn.compact
  def __call__(
      self, image: jnp.ndarray, context_input: jnp.ndarray, *, train: bool
  ):
    """Tokenizes the image using an EfficientNet.

    Args:
      image: jnp.Array with batch and seqlen leading dimensions. We assume the
        input image is of size 300x300, since the EfficientNet takes in images
        of that size.
      context_input: jnp.Array with shape (batch * seqlen, size).
      train: Training mode.

    Returns:
      shape (batch, seqlen, num_tokens, num_features) array.
    """
    bs, seqlen, *_ = image.shape

    # The efficientnet-b3 model uses 300x300 images.
    efficientnet_config = efficientnet.MODEL_CONFIGS['efficientnet-b3']
    image = jnp.reshape(image, [bs * seqlen, 300, 300, 3])
    image -= jnp.array(efficientnet.MEAN_RGB)
    image /= jnp.array(efficientnet.STDDEV_RGB)

    # Apply film in EfficientNet.
    x = efficientnet.EfficientNetWithFilm(efficientnet_config)(
        image, context_input=context_input, train=train
    )

    # 1x1 conv. This corresponds to the 1x1 conv here:
    # google-research/robotics_transformer/film_efficientnet/pretrained_efficientnet_encoder.py
    var_init = nn.initializers.variance_scaling(
        scale=1.0,
        mode='fan_in',
        distribution='truncated_normal',
    )
    x = nn.Conv(
        features=self.num_features,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_init=var_init,
    )(x)

    x = film_conditioning.FilmConditioning(num_channels=self.num_features)(
        x, context_input
    )

    if self.use_token_learner:
      x = token_learner.TokenLearnerModuleV11(num_tokens=self.num_tokens)(
          x, deterministic=not train
      )

    x = jnp.reshape(x, [bs, seqlen, self.num_tokens, -1])

    return x


class TokenLearnerModuleV11(nn.Module):
  """TokenLearner module Version 1.1, using slightly different conv. layers.

  Instead of using 4 conv. layers with small channels to implement spatial
  attention, this version uses a MLP with gelu inbetween. It also uses softmax
  instead of sigmoid. We confirmed that this version works better in general.

  From google-research/scenic/projects/token_learner/model.py.

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

    selected = attention_layers.MlpBlock(
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


def tokenize_action(
    actions: Dict[str, jnp.ndarray],
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
  """Tokenizes the action for the RT-1 task.

  <name>: <shape> <bounds>
  terminate_episode: (3,) int32,
    mode 0: terminate episode
    mode 1: arm + gripper

    mode 2: base
  world_vector: (3,) [-1.0, 1.0] (RT-1) or [-2.0, 2.0] (RT-1-X)
  rotation_delta: (3,) [-np.pi, np.pi]
  gripper_closedness_action: (1,) [-1, 1]
  base_displacement_vertical_rotation: (1,) [-np.pi, np.pi]
  base_displacement_vector: (2,) [-1.0, 1.0]

  Args:
    actions: The raw action dictionary.
    vocab_size: The vocab size of the tokenized actions.
    world_vector_range: The bounds to use for the world_vector token.

  Returns:
    the tokenized action.
  """
  action_tokens = []

  # Handle the discrete one first.
  terminate_episode = actions['terminate_episode']
  terminate_episode = jnp.argmax(terminate_episode, axis=-1)
  terminate_episode = jnp.expand_dims(terminate_episode, -1)
  terminate_episode = terminate_episode.astype(jnp.int32)
  action_tokens.append(terminate_episode)

  for act_name, act_min, act_max in [
      ('world_vector', world_vector_range[0], world_vector_range[1]),
      ('rotation_delta', -np.pi / 2, np.pi / 2),
      ('gripper_closedness_action', -1.0, 1.0),
      ('base_displacement_vertical_rotation', -np.pi, np.pi),
      ('base_displacement_vector', -1.0, 1.0),
  ]:
    act = actions[act_name]
    act = jnp.clip(act, act_min, act_max)
    act = (act - act_min) / (act_max - act_min)
    act = act * (vocab_size - 1)
    act = act.astype(jnp.int32)
    action_tokens.append(act)

  tokenized = jnp.concatenate(action_tokens, axis=-1)
  return tokenized


def detokenize_action(
    tokenized_actions: jnp.ndarray,
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, jnp.ndarray]:
  """De-tokenizes the action for the RT-1 task.

  See `tokenize_action` for information on the action structure.

  Args:
    tokenized_actions: The tokenized action vector.
    vocab_size: The vocab size of the tokenized actions.
    world_vector_range: The bounds to use for the world_vector token.

  Returns:
    the detokenized action dictionary.
  """
  terminate_episode = tokenized_actions[:, 0]
  terminate_episode = jax.nn.one_hot(terminate_episode, 3)

  raw_actions = dict(
      world_vector=tokenized_actions[:, 1:4].astype(jnp.float32),
      rotation_delta=tokenized_actions[:, 4:7].astype(jnp.float32),
      gripper_closedness_action=tokenized_actions[:, 7:8].astype(jnp.float32),
      base_displacement_vertical_rotation=tokenized_actions[:, 8:9].astype(
          jnp.float32
      ),
      base_displacement_vector=tokenized_actions[:, 9:11].astype(jnp.float32),
  )

  act_dict = {'terminate_episode': terminate_episode.astype(jnp.int32)}
  for act_name, act_min, act_max in [
      ('world_vector', world_vector_range[0], world_vector_range[1]),
      ('rotation_delta', -np.pi / 2, np.pi / 2),
      ('gripper_closedness_action', -1.0, 1.0),
      ('base_displacement_vertical_rotation', -np.pi, np.pi),
      ('base_displacement_vector', -1.0, 1.0),
  ]:
    act = raw_actions[act_name]
    act = act / (vocab_size - 1)
    act = act * (act_max - act_min)
    act = act + act_min
    act_dict[act_name] = act

  return act_dict


class RT1(nn.Module):
  """Full RT-1 / RT-1-X architecture."""

  num_layers: int = 8
  layer_size: int = 128
  num_heads: int = 8
  feed_forward_hidden_size: int = 512
  feed_forward_output_size: int = 512
  ffn_option: FFNOptions = FFNOptions.SWIGLU
  dropout_rate: float = 0.1
  vocab_size: int = 256
  num_image_tokens: int = 8
  num_action_tokens: int = 11
  image_num_features: int = 512

  world_vector_range: Tuple[float, float] = (-1.0, 1.0)

  use_token_learner: bool = True

  # By default, mask out previous actions.
  include_prev_timesteps_actions: bool = False

  sow_intermediates: bool = False

  def setup(self):
    self.image_tokenizer = ImageTokenizer(
        num_tokens=self.num_image_tokens,
        num_features=self.image_num_features,
        use_token_learner=self.use_token_learner,
    )

  def tokenize_image(
      self, image: jnp.ndarray, context: jnp.ndarray, *, train: bool
  ):
    bs, seqlen, *_ = image.shape
    context = jnp.reshape(context, [bs * seqlen, -1])
    return self.image_tokenizer(image, context_input=context, train=train)

  @nn.compact
  def __call__(
      self,
      obs: Dict[str, jnp.ndarray],
      act: Dict[str, jnp.ndarray],
      obs_tokens: Optional[jnp.ndarray] = None,
      act_tokens: Optional[jnp.ndarray] = None,
      *,
      train: bool,
  ):
    bs = obs['image'].shape[0]
    seqlen = obs['image'].shape[1]

    # Depending on whether `obs_tokens` is passed, we either run the full
    # sequence of images through the image tokenizer, or simply use the
    # image tokens passed into this function. `obs_tokens` is usually passed
    # during an inference call when caching tokens from previous elements of
    # the input sequence.
    if obs_tokens is None:
      # Get image + language fused tokens.
      image = obs['image']
      lang = obs['natural_language_embedding']
      lang = jnp.reshape(lang, [bs * seqlen, -1])
      context_image_tokens = self.image_tokenizer(
          image=image, context_input=lang, train=train
      )
    else:
      context_image_tokens = obs_tokens

    if self.sow_intermediates:
      self.sow('intermediates', 'image_tokens', context_image_tokens)

    # We either tokenize the action ourselves using `tokenize_action_fn` or
    # use the tokens passed into this function. `act_tokens` is usually supplied
    # during an inference call when caching tokens from previous actions.
    if act_tokens is None:
      action_tokens = tokenize_action(
          act, self.vocab_size, self.world_vector_range
      )  # pylint: disable=too-many-function-args
    else:
      action_tokens = act_tokens

    if self.include_prev_timesteps_actions:
      # Always zero out the final action tokens.
      previous_action_tokens = action_tokens[:, : (seqlen - 1), :]
      zero_action_tokens = jnp.zeros((bs, 1, self.num_action_tokens))
      action_tokens = jnp.concatenate(
          [previous_action_tokens, zero_action_tokens], axis=-2
      )

      # Project the actions to the token dimension.
      action_tokens = jax.nn.one_hot(action_tokens, num_classes=self.vocab_size)
      action_tokens = nn.Dense(self.image_num_features)(action_tokens)
    else:
      # If we're not including the previous actions, then we can zero out
      # the action tokens. We do it here to ensure tokens are consistently
      # zero regardless of the input actions passed to the function.
      action_tokens = jnp.zeros(
          (bs, seqlen, self.num_action_tokens, self.image_num_features)
      )

    # Assemble the input tokens into a single sequence.
    full_tokens = jnp.concatenate(
        [context_image_tokens, action_tokens], axis=-2
    )

    num_action_tokens = action_tokens.shape[-2]
    full_tokens = jnp.reshape(
        full_tokens,
        [bs, seqlen * (self.num_image_tokens + num_action_tokens), -1],
    )

    attn_mask = self._construct_attn_mask(
        seqlen * (self.num_image_tokens + self.num_action_tokens)
    )
    output_tokens = Transformer(
        num_layers=self.num_layers,
        layer_size=self.layer_size,
        num_heads=self.num_heads,
        feed_forward_hidden_size=self.feed_forward_hidden_size,
        feed_forward_output_size=self.feed_forward_output_size,
        dropout_rate=self.dropout_rate,
        vocab_size=self.vocab_size,
        ffn_option=self.ffn_option,
    )(full_tokens, attn_mask=attn_mask, train=train)

    return output_tokens

  def _get_action_index_for_token(self, k: int, num_tokens: int):
    """Returns action associated with the token at given position `k`.

    If k is not an action token then it returns -1.
    If k is part of the first action in the sequence then returns 0 etc.

    Based on `_get_action_index_for_token` here:
    google-research/robotics_transformer/transformer_network.py

    Args:
      k: an int that represents the position in the sequence.
      num_tokens: The total number of tokens in the sequence.

    Returns:
      The index of the action that this position belongs to, or if this
      position is part of an image token then returns -1.
    """
    if k < 0 or k >= num_tokens:
      return -1

    single_time_step_num_tokens = self.num_image_tokens + self.num_action_tokens
    n = k
    if n % single_time_step_num_tokens < self.num_image_tokens:
      return -1

    return int(n / single_time_step_num_tokens)

  def _construct_attn_mask(self, num_tokens: ...):
    """Generate mask for action prediction loss.

    This masks out all action tokens.

    Based on `_generate_masks` here:
    google-research/robotics_transformer/transformer_network.py

    Args:
      num_tokens: The number of tokens with which to construct the input mask.

    Returns:
      A (num_tokens, num_tokens) attention mask.
    """
    default_attn_mask = np.tril(np.ones((num_tokens, num_tokens), np.int32))
    action_mask = np.zeros(shape=(num_tokens, num_tokens), dtype=np.int32)

    for i in range(num_tokens):
      for j in range(num_tokens):
        action_i = self._get_action_index_for_token(i, num_tokens)
        action_j = self._get_action_index_for_token(j, num_tokens)
        mask = 0
        if action_i != -1 and action_j != -1:
          # Ignore actions of previous steps.
          if action_j < action_i:
            mask = 1
          # If we're not auto-regression, ignore action dimensions of current
          # step.
          if action_j == action_i and j <= i:
            mask = 1
        # i not is an action, but j is an action token.
        # Hence, also mask j when predicting i, to prevent accidental
        # dependency between output and masked dimensions because the output
        # can still depend on the masked dimensions when predictions of the
        # transformer layers after the first layer depends on the masked
        # dimensions.
        elif action_j != -1:
          if not self.include_prev_timesteps_actions and j < i:
            mask = 1
        action_mask[i, j] = mask
    return default_attn_mask - action_mask

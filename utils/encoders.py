# Source: https://github.com/nakamotoo/action_chunk_q_learning/tree/main
import functools
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import default_init
from transformers import AutoImageProcessor, FlaxResNetModel

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class ResNet18Pretrain(nn.Module):
    mlp_hidden_dims: tuple = (1000,)

    def __init__(self):        
        self._RESNET_MODEL_NAME = "microsoft/resnet-18"
        self._RESNET_MODEL = FlaxResNetModel.from_pretrained(self._RESNET_MODEL_NAME)
        self._RESNET_PROCESSOR_CONFIG = AutoImageProcessor.from_pretrained(self._RESNET_MODEL_NAME)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        _ = self.param('dummy', nn.initializers.zeros, (1,))
        original_shape = observations.shape
        was_chunked = len(original_shape) == 5
        was_unbatched = len(original_shape) == 3

        if was_chunked:
            # Input is (B, T, H, W, C). Merge B and T dimensions.
            b, t, h, w, c = original_shape
            x = observations.reshape(b * t, h, w, c)
        elif was_unbatched:
            # Input is (H, W, C). Add a temporary batch dimension.
            x = jnp.expand_dims(observations, axis=0)
        else:
            # Input is already the expected (B, H, W, C).
            x = observations
            
        effective_batch_size = x.shape[0]

        target_size = self._RESNET_PROCESSOR_CONFIG.size['shortest_edge']
        target_h = target_size
        target_w = target_size
        
        x = x.astype(jnp.float32)
        x = jax.image.resize(x, (x.shape[0], target_h, target_w, x.shape[-1]), method='bilinear')
        
        mean = jnp.array(self._RESNET_PROCESSOR_CONFIG.image_mean)
        std = jnp.array(self._RESNET_PROCESSOR_CONFIG.image_std)
        x = (x / 255.0 - mean) / std
        
        x = jnp.transpose(x, (0, 3, 1, 2))
        outputs = self._RESNET_MODEL(pixel_values=x, params=self._RESNET_MODEL.params, train=False)
        features = outputs.pooler_output
        features = features.reshape(effective_batch_size, -1)

        if was_chunked:
            feature_dim = features.shape[-1]
            features = features.reshape(b, t, feature_dim)
        elif was_unbatched:
            features = jnp.squeeze(features, axis=0)
        return jax.lax.stop_gradient(features)

class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))
    
        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'
    mlp_hidden_dims: Sequence[int] = (50,)
    layer_norm: bool = False    

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
        out = x.reshape((*x.shape[:-3], -1))
        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=True, activations=nn.tanh)(out)
        return out


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'small': Encoder,
    'resnet18_frozen': ResNet18Pretrain,
}
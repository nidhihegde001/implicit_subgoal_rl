# Source: https://github.com/nakamotoo/action_chunk_q_learning/tree/main
from typing import Any, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax

def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def transformer_init(gain=1.0):
    return nn.initializers.orthogonal(gain)

def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls, variable_axes={'params': 0}, split_rngs={'params': True, 'dropout': True},
        in_axes=in_axes, out_axes=out_axes, axis_size=num_qs, **kwargs
    )

def sinusoidal_pos_encoding(max_len: int, d_model: int) -> jnp.ndarray:
    position = jnp.arange(max_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[jnp.newaxis, ...]

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x): 
        return x

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
        
        # print("MLP Out shape: ", x.shape)
        return x

class MLPCond(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, cond):
        cond_processed = nn.Dense(self.hidden_dims[0] // 4, kernel_init=self.kernel_init)(cond)
        
        for i, size in enumerate(self.hidden_dims):
            if i == 0:
                x = nn.Dense(size, kernel_init=self.kernel_init)(
                    jnp.concatenate([x, cond], axis=-1)
                )
            else:
                x = nn.Dense(size, kernel_init=self.kernel_init)(x)
                if i < len(self.hidden_dims) - 1:
                    gate = nn.Dense(size, kernel_init=self.kernel_init)(cond_processed)
                    x = x * nn.sigmoid(gate)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class AdaLNTransformerBlock(nn.Module):
    """A Transformer block with AdaLN-Zero conditioning."""
    d_model: int
    n_heads: int
    dropout: float = 0.1
    kernel_init: Any = transformer_init(gain=1.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, deterministic: bool = True):
        # Predict modulation parameters from the conditioning signal
        modulation = nn.silu(cond)
        modulation = nn.Dense(6 * self.d_model, kernel_init=nn.initializers.zeros)(modulation)
        
        # Ensure correct broadcasting shape
        if modulation.ndim == 2 and x.ndim == 3:
            modulation = modulation[:, None, :]  # Add sequence dimension
            
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-Attention with AdaLN
        x_norm1 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa
        
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, 
            kernel_init=self.kernel_init, 
            dropout_rate=self.dropout
        )(inputs_q=x_modulated1, deterministic=deterministic)
        
        x = x + gate_msa * attn_out

        # MLP with AdaLN
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp

        mlp_out = nn.Dense(4 * self.d_model, kernel_init=self.kernel_init)(x_modulated2)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dropout(self.dropout)(mlp_out, deterministic=deterministic)
        mlp_out = nn.Dense(self.d_model, kernel_init=nn.initializers.zeros)(mlp_out)
        
        x = x + gate_mlp * mlp_out
        return x
    
class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.1
    kernel_init: Any = transformer_init(gain=0.02)

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, deterministic: bool = True):
        # Fix the array truth value issue - use explicit None check
        kv_input = x if context is None else context
        
        attn_kwargs = {
            'num_heads': self.n_heads, 
            'kernel_init': self.kernel_init, 
            'dropout_rate': self.dropout
        }
        attn_out = nn.MultiHeadDotProductAttention(**attn_kwargs)(
            inputs_q=x, inputs_kv=kv_input, deterministic=deterministic
        )
        x = nn.LayerNorm()(x + attn_out)
        
        # Feed-forward network
        mlp_out = nn.Dense(4 * self.d_model, kernel_init=self.kernel_init)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dropout(self.dropout)(mlp_out, deterministic=deterministic)
        mlp_out = nn.Dense(self.d_model, kernel_init=self.kernel_init)(mlp_out)
        mlp_out = nn.Dropout(self.dropout)(mlp_out, deterministic=deterministic)
        x = nn.LayerNorm()(x + mlp_out)
        return x

class SingleTransformerCritic(nn.Module):
    """Single transformer critic for cross-attention between actions and states."""
    hidden_dim: int
    n_layers: int
    n_heads: int
    dropout: float
    output_dim: int = 1

    @nn.compact
    def __call__(self, action_query: jnp.ndarray, state_context: jnp.ndarray, deterministic: bool = True):
        x = action_query
        for i in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout,
                name=f"block_{i}"
            )(x, context=state_context, deterministic=deterministic)

        # Output head
        head_in = x.squeeze(axis=1)
        q_out = nn.LayerNorm()(head_in)
        q_out = nn.Dense(self.hidden_dim // 2, kernel_init=transformer_init(gain=0.02))(q_out)
        q_out = nn.gelu(q_out)
        q_out = nn.Dense(self.output_dim, kernel_init=transformer_init(gain=0.02))(q_out)
        return q_out

def ensemblize_critic(num_ensembles: int, **critic_kwargs):
    """Create an ensemble of critics using vmap, handling deterministic properly."""
    
    class EnsembledCritic(nn.Module):
        @nn.compact
        def __call__(self, action_query, state_context, deterministic: bool = True):
            # Create vmapped version without passing deterministic as kwarg to avoid warning
            vmapped_critic = nn.vmap(
                SingleTransformerCritic,
                variable_axes={'params': 0},
                split_rngs={'params': True, 'dropout': True},
                in_axes=(None, None, None),  # action_query, state_context, deterministic
                out_axes=0,
                axis_size=num_ensembles
            )(**critic_kwargs)
            
            # Call vmapped critics with deterministic as positional argument
            return vmapped_critic(action_query, state_context, deterministic)
    
    return EnsembledCritic()

class LogParam(nn.Module):
    """Scalar parameter module with log scale."""
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)

class NoiseInjectionNetwork(nn.Module):
    """Noise injection network for ReinFlow."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    state_dim: int = 128  # Input state dimension
    layer_norm: bool = True
    min_noise_std: float = 0.001
    max_noise_std: float = 1.0
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, time_steps: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            observations: (B, S) state observations
            time_steps: (B, 1) time values in [0, 1]
        Returns:
            (B,) noise standard deviations
        """
        x = jnp.concatenate([observations, time_steps], axis=-1)
        for size in self.hidden_dims:
            x = nn.Dense(features=size)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        
        # Output scalar noise for simplicity (same noise for all action dims)
        # You could output (B, action_dim) for dimension-specific noise
        logvar = nn.Dense(features=1)(x)
        logvar = jnp.tanh(logvar)
        
        logvar_min = jnp.log(self.min_noise_std ** 2)
        logvar_max = jnp.log(self.max_noise_std ** 2)
        logvar = logvar_min + (logvar_max - logvar_min) * (logvar + 1.0) / 2.0
        noise_std = jnp.exp(0.5 * logvar).squeeze(-1)  # (B,)
        
        return noise_std

class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""
    def mode(self):
        return self.bijector.forward(self.distribution.mode())

class Actor(nn.Module):
    """Gaussian actor network."""
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = True
    const_std: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution
    
class GenerativeTransformerPolicy(nn.Module):
    """Corrected AdaLN Transformer for denoising/flow matching policies."""
    action_dim: int
    horizon_length: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    dropout: float = 0.1
    encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, 
                 time_conditioning: jnp.ndarray, is_encoded: bool = False, 
                 deterministic: bool = True):
        """
        Args:
            observations: (B, S) state observations
            actions: (B, H, A) action sequence (noisy for diffusion, interpolated for flow)
            time_conditioning: (B, 1) or (B,) time/noise level
            is_encoded: whether observations are already encoded
            deterministic: for dropout
        Returns:
            (B, H, A) predicted noise (diffusion) or velocity (flow)
        """
        B, H, A = actions.shape
        assert H == self.horizon_length
        assert A == self.action_dim

        encoder_module = self.encoder or Identity()
        state_emb = encoder_module(observations) if not is_encoded else observations
        
        if time_conditioning.ndim == 1:
            time_conditioning = time_conditioning[:, None]
        
        # Project state to hidden dim
        state_proj = nn.Dense(self.hidden_dim, name="state_proj")(state_emb)
        
        # Project time to hidden dim  
        time_emb = nn.Dense(self.hidden_dim, name="time_embed")(time_conditioning)
        
        # Combine state and time for AdaLN conditioning
        # Use concatenation and another projection
        cond_combined = jnp.concatenate([state_proj, time_emb], axis=-1)
        cond_emb = nn.Dense(self.hidden_dim, name="cond_proj")(cond_combined)
        
        # Embed action sequence
        x = nn.Dense(self.hidden_dim, name="action_embed")(actions)
        
        # Add positional encoding
        x = x + sinusoidal_pos_encoding(self.horizon_length, self.hidden_dim)
        
        # Process through AdaLN transformer blocks
        # No causal mask - all positions attend to all positions
        for i in range(self.n_layers):
            x = AdaLNTransformerBlock(
                d_model=self.hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout,
                name=f"adaln_block_{i}"
            )(x, cond_emb, deterministic=deterministic)
        
        # Output projection
        x = nn.LayerNorm()(x)
        output = nn.Dense(self.action_dim, kernel_init=default_init(0.02), name="output_head")(x)
        
        return output

class ActorVectorField(nn.Module):
    action_dim: int
    horizon_length: int
    encoder: Optional[nn.Module] = None
    
    # --- Backend configuration ---
    use_transformer: bool = True
    
    # Transformer-specific parameters
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    
    # MLP-specific parameters
    mlp_hidden_dims: Sequence[int] = (512, 256, 128)
    layer_norm_mlp: bool = True

    def setup(self):
        """Initializes the selected backbone network."""
        if self.use_transformer:
            self.backbone = GenerativeTransformerPolicy(
                action_dim=self.action_dim,
                horizon_length=self.horizon_length,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
                encoder=self.encoder
            )
        else:
            output_dim = self.action_dim * self.horizon_length
            mlp_dims = list(self.mlp_hidden_dims) + [output_dim]
            self.backbone = MLPCond(
                hidden_dims=mlp_dims,
                layer_norm=self.layer_norm_mlp
            )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, times: jnp.ndarray, **kwargs):
        B = actions.shape[0]
        is_encoded = kwargs.get('is_encoded', False)
        
        if self.encoder is not None:
            state_emb = self.encoder(observations) if not is_encoded else observations
        else:
            state_emb = observations
        
        if times.ndim == 1:
            times = times[:, None]

        if self.use_transformer:
            actions_seq = actions.reshape(B, self.horizon_length, self.action_dim)
            # Pass state_emb explicitly and set is_encoded=True for the transformer backbone
            kwargs['is_encoded'] = True 
            velocities_seq = self.backbone(state_emb, actions_seq, times, **kwargs)
            return velocities_seq.reshape(B, -1)
        else:
            cond = jnp.concatenate([state_emb, times], axis=-1)
            return self.backbone(actions, cond)

class DenoisingActor(nn.Module):
    """DDPM/DDIM policy that supports both Transformer and MLP backends."""
    action_dim: int
    horizon_length: int
    encoder: Optional[nn.Module] = None
    
    # --- Backend configuration ---
    use_transformer: bool = True
    
    # Transformer-specific parameters
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    
    # MLP-specific parameters
    mlp_hidden_dims: Sequence[int] = (512, 256, 128)
    layer_norm_mlp: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, noisy_actions: jnp.ndarray, 
                 timesteps_k: jnp.ndarray, is_encoded: bool = False, 
                 deterministic: bool = True):
        B = noisy_actions.shape[0]
        
        if self.use_transformer:
            # Reshape actions for the Transformer
            actions_seq = noisy_actions.reshape(B, self.horizon_length, self.action_dim)
            
            # Transformer backbone for predicting noise
            noise_pred_seq = GenerativeTransformerPolicy(
                action_dim=self.action_dim,
                horizon_length=self.horizon_length,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
                encoder=self.encoder
            )(observations, actions_seq, timesteps_k, is_encoded=is_encoded, deterministic=deterministic)
            
            return noise_pred_seq.reshape(B, -1)
        else:
            # MLP backbone for predicting noise
            encoder_module = self.encoder or Identity()
            state_emb = encoder_module(observations) if not is_encoded else observations
            
            if timesteps_k.ndim == 1:
                timesteps_k = timesteps_k[:, None]
                
            cond_input = jnp.concatenate([state_emb, timesteps_k], axis=-1)
            
            output_dim = self.action_dim * self.horizon_length
            mlp_dims = list(self.mlp_hidden_dims) + [output_dim]
            
            noise_pred = MLPCond(
                hidden_dims=mlp_dims,
                layer_norm=self.layer_norm_mlp
            )(noisy_actions, cond_input)
            
            return noise_pred

class SingleAdaLNCritic(nn.Module):
    """Single AdaLN-Zero Transformer critic."""
    action_dim: int
    horizon_length: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    dropout: float
    ofn: bool = False  # Whether to use OFN normalization
    output_dim: int = 1

    @nn.compact
    def __call__(self, actions: jnp.ndarray, state_cond: jnp.ndarray, 
                 deterministic: bool = True):
        """
        Args:
            actions: (B, H, A) action sequence
            state_cond: (B, hidden_dim) state conditioning
        Returns:
            (B, 1) Q-values
        """
        B, H, A = actions.shape
        
        # Embed actions
        x = nn.Dense(self.hidden_dim, name="action_embed")(actions)
        
        # Add positional encoding
        x = x + sinusoidal_pos_encoding(self.horizon_length, self.hidden_dim)

        # Process with AdaLN blocks
        for i in range(self.n_layers):
            x = AdaLNTransformerBlock(
                d_model=self.hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout,
                name=f"adaln_block_{i}"
            )(x, state_cond, deterministic=deterministic)
        
        # Pool and output
        pooled = jnp.mean(x, axis=1)  # Average pooling over sequence
        q_out = nn.LayerNorm()(pooled)
        q_out = nn.Dense(self.hidden_dim // 2, kernel_init=transformer_init(gain=1.0))(q_out)
        q_out = nn.gelu(q_out)
        if self.ofn:
            q_out = q_out / jnp.linalg.norm(q_out, ord=2, axis=-1, keepdims=True)
        q_out = nn.Dense(self.output_dim, kernel_init=transformer_init(gain=1.0))(q_out)
        
        return q_out

class Value(nn.Module):
    action_dim: int
    horizon_length: int
    num_ensembles: int = 2
    critic_loss_type: str = 'mse'  # 'mse' or 'hlgauss'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    encoder: Optional[nn.Module] = None
    
    use_transformer: bool = True
    
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    ofn: bool = False
    
    mlp_hidden_dims: Sequence[int] = (512, 256, 128)
    layer_norm_mlp: bool = True
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, is_encoded: bool = False, deterministic: bool = True, return_logits: bool = False):
        B = actions.shape[0]
        encoder_module = self.encoder or Identity()
        states = encoder_module(observations) if not is_encoded else observations
        
        if self.use_transformer:
            actions_seq = actions.reshape(B, self.horizon_length, self.action_dim)
            state_cond = nn.Dense(self.hidden_dim, name="state_cond_embed")(states)
            
            vmapped_critic = nn.vmap(
                SingleAdaLNCritic,
                variable_axes={'params': 0},
                split_rngs={'params': True, 'dropout': True},
                in_axes=(None, None, None),
                out_axes=0,
                axis_size=self.num_ensembles
            )(
                action_dim=self.action_dim,
                horizon_length=self.horizon_length,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
                ofn=self.ofn
            )
            q_values = vmapped_critic(actions_seq, state_cond, deterministic)

        else:
            mlp_input = jnp.concatenate([states, actions], axis=-1)
            EnsembledMLPCritic = ensemblize(MLP, num_qs=self.num_ensembles)
            
            critic_mlp_dims = list(self.mlp_hidden_dims) + [1 if self.critic_loss_type == 'mse' else self.num_bins]
            q_values = EnsembledMLPCritic(
                hidden_dims=critic_mlp_dims,
                layer_norm=self.layer_norm_mlp
            )(mlp_input)
            # print("q values from critic: ", q_values.shape)

        if self.critic_loss_type == 'hlgauss':
            q = jnp.sum(
                jax.nn.softmax(q_values)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )
            if return_logits:
                return q, q_values
            return q
        else:
            return q_values.squeeze(-1)

class StateValue(nn.Module):
    """
    A state-value function V(s) network for PPO-based agents like DPPO.
    """
    encoder: Optional[nn.Module] = None
    mlp_hidden_dims: Sequence[int] = (256, 256)
    layer_norm_mlp: bool = True
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, is_encoded: bool = False):
        encoder_module = self.encoder or Identity()
        states = encoder_module(observations) if not is_encoded else observations
        
        value_mlp_dims = list(self.mlp_hidden_dims) + [1]
        v_values = MLP(
            hidden_dims=value_mlp_dims,
            layer_norm=self.layer_norm_mlp
        )(states)
        
        return v_values.squeeze(-1)
        
# Backward compatibility aliases
TransformerActor = ActorVectorField  # Keep old name for compatibility
# Modified from https://github.com/nakamotoo/action_chunk_q_learning/blob/main/agents/sac.py

import copy
from typing import Any, Dict

import flax
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
print("JAX devices:", jax.devices())
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, LogParam


class MultiGoalSACAgent(flax.struct.PyTreeNode):
    """
    Multi-goal SAC agent that trains separate critic networks for each sub-goal
    while sharing a single actor and temperature parameter.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    n_goals: int = nonpytree_field()

    def _select_actor_actions_and_logp(self, observations, rng):
        """Sample actions and log-probs from the actor."""
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(observations)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        return actions, log_probs, rng

    def _compute_target_q_for_goal(self, goal_idx, next_obs, next_actions, next_log_probs):
        """
        Compute next Q (aggregated across ensemble) for target critic of a single goal.
        returns shape (batch,)
        """
        target_name = f"target_critic_goal_{goal_idx}"
        # returns shape (num_qs, batch)
        next_qs = self.network.select(target_name)(next_obs, actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)
        return next_q

    def critic_loss(self, batch, grad_params, rng):
        """
        Compute critic losses for each goal and sum them (or return dict).
        Each critic is trained only on samples whose `phases == goal_idx`.
        """
        info: Dict[str, Any] = {}
        B = batch['states'].shape[0]
        obs = batch['states']
        actions = batch['actions']
        next_obs = batch['next_states']
        dones = batch.get('dones', None)
        discount = self.config['discount']
        subgoal_rewards = batch['subgoal_rewards']

        # flatten actions if chunking like reference (keeps compatibility)
        # if self.config["action_chunking"]:
        #     batch_actions = jnp.reshape(actions, (B, -1))
        # else:
        # batch_actions = actions[..., 0, :]

        # Sample next actions & log probs using current actor
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(next_obs)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        # Temperature
        temp = self.network.select('temp')()

        total_critic_loss = 0.0
        per_goal_losses = []

        # For all goals compute target q and critic loss masked by phase
        for g in range(self.n_goals):
            # Targets use subgoal-specific reward
            # batch['subgoal_rewards'] shape is (B, G)
            r_g = subgoal_rewards[:, g]

            # compute next_q for this goal from its target critic
            next_q_g = self._compute_target_q_for_goal(g, next_obs, next_actions, next_log_probs)

            # optionally include entropy term in target
            if self.config.get('entropy_backup', False):
                target_q_g = r_g + discount * (1.0 - dones) * (next_q_g - temp * next_log_probs)
            else:
                target_q_g = r_g + discount * (1.0 - dones) * next_q_g

            # current q (num_qs, batch)
            critic_name = f"critic_goal_{g}"
            qs = self.network.select(critic_name)(obs, actions=actions, params=grad_params)
            if self.config['q_agg'] == 'min':
                q_agg = jnp.min(qs, axis=0)
            else:
                q_agg = jnp.mean(qs, axis=0)

            # mask = 1 for samples whose phase == g, else 0
            phase_mask = (batch['phases'] == g).astype(jnp.float32)  

            # MSE between q_agg and target, only accumulate where phase_mask==1
            mse = (q_agg - target_q_g) ** 2
            # eps = 1e-8
            # loss_g = jnp.sum(mse * phase_mask)/(jnp.sum(phase_mask) + eps) # remove this (use to train all critics)
            loss_g = jnp.mean(mse)

            per_goal_losses.append(loss_g)
            total_critic_loss = total_critic_loss + loss_g

            # logging
            info[f'critic/g{g}/loss'] = loss_g
            info[f'critic/g{g}/q_mean'] = q_agg.mean()
            info[f'critic/g{g}/target_q_mean'] = target_q_g.mean()
            info[f'critic/_phase_g{g}_num_datapoints'] = phase_mask.sum()

        total_critic_loss = total_critic_loss / self.n_goals
        info['critic/avg_loss'] = total_critic_loss
        return total_critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """
        Actor loss: for each sample, the Q term is taken from the critic corresponding
        to the sample's phase. We compute actions from actor, get their log-probs,
        and evaluate all goal-critics to pick the per-sample critic by phase.
        """
        info: Dict[str, Any] = {}
        obs = batch['states']
        B = obs.shape[0]

        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(obs, params=grad_params)
        sampled_actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        # print("sampled actions shape: ", sampled_actions.shape)
        act_flat = sampled_actions

        # # flatten if needed
        # if self.config['action_chunking']:
        #     act_flat = jnp.reshape(sampled_actions, (B, -1))
        # else:
        #     act_flat = sampled_actions[..., 0, :]

        # Evaluate each goal critic on (obs, act_flat)
        # Collect aggregated Qs per goal into q_vals_of_goals: shape (G, B)
        q_vals_per_goal = []
        for g in range(self.n_goals):
            critic_name = f"critic_goal_{g}"
            qs = self.network.select(critic_name)(obs, actions=act_flat, params=grad_params)
            if self.config['q_agg'] == 'min':
                q_agg = jnp.min(qs, axis=0)
            else:
                q_agg = jnp.mean(qs, axis=0)
            q_vals_per_goal.append(q_agg)
        q_vals_per_goal = jnp.stack(q_vals_per_goal, axis=0)  # (G, B)

        # Now pick per-sample q according to phase: q_per_sample[b] = q_vals_per_goal[phase[b], b]
        idxs = jnp.arange(B)
        phases = batch['phases'].astype(jnp.int32)
        q_per_sample = q_vals_per_goal[phases, idxs]

        temp = self.network.select('temp')()
        actor_loss = jnp.mean((temp * log_probs - q_per_sample) * batch.get('valid', 1.0))

        # Temperature loss (auto temp tuning)
        temp_param = self.network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs * batch.get('valid', 1.0)).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + temp_loss

        info.update({
            'actor/actor_loss': actor_loss,
            'actor/temp_loss': temp_loss,
            'actor/log_probs_mean': log_probs.mean(),
            'actor/entropy': -log_probs.mean(),
            'actor/q_mean': q_per_sample.mean(),
            'actor/temp': temp,
        })
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info: Dict[str, Any] = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss_val, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss_val, actor_info = self.actor_loss(batch, grad_params, actor_rng)

        # merge infos
        for k, v in critic_info.items():
            info[k] = v
        for k, v in actor_info.items():
            info[k] = v

        loss = critic_loss_val + actor_loss_val
        return loss, info

    def _soft_update_all_targets(self, network):
        """Soft update all goals' critic target parameters."""
        params = unfreeze(network.params)
        for goal_idx in range(self.n_goals):
            target_name = f'modules_target_critic_goal_{goal_idx}'
            src_name = f'modules_critic_goal_{goal_idx}'
            params[target_name] = jax.tree_util.tree_map(
                lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
                self.network.params[src_name],
                self.network.params[target_name],
            )
        return network.replace(params=freeze(params))

    @staticmethod
    def _update(agent, batch):
        """Single-step update: minimize total loss w.r.t. network params (actor + all critics + temp)."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        # Soft-update all goal target critics
        new_network = agent._soft_update_all_targets(new_network)

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Batch update using jax.lax.scan if batch is structured as sequence of batches."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None, deterministic=False):
        dist = self.network.select('actor')(observations)
        rng = rng if rng is not None else self.rng
        if deterministic:
            if hasattr(dist, 'mode'):
                return dist.mode()
            else:
                return dist.mean()
        else:
            return dist.sample(seed=rng)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, n_goals=3):
        """Construct the agent with per-goal critics/targets."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        print(f"ex obs shape: {ob_dims}, ex action shape: {action_dim}")
        # if config["action_chunking"]:
        #     full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        # else:
        full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Create per-goal critic defs
        critic_defs = {}
        critic_args = {}
        print("encoder: ", encoders.get('critic'))
        for g in range(n_goals):
            critic_def = Value(
                action_dim=full_action_dim,
                horizon_length=1,
                mlp_hidden_dims=config['value_hidden_dims'],
                layer_norm_mlp=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config.get('num_bins', None),
                q_min=config.get('q_min', None),
                q_max=config.get('q_max', None),
                use_transformer=False,
            )
            critic_defs[f'critic_goal_{g}'] = critic_def
            critic_args[f'critic_goal_{g}'] = (ex_observations, full_actions)

            # target critic
            critic_defs[f'target_critic_goal_{g}'] = copy.deepcopy(critic_def)
            critic_args[f'target_critic_goal_{g}'] = (ex_observations, full_actions)

        # Actor + temp
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,
            state_dependent_std=True,
        )
        temp_def = LogParam(init_value=config.get('init_temperature', 1.0))

        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            temp=(temp_def, ()),
            **{k: (v, critic_args[k]) for k, v in critic_defs.items()}
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        
        # Copy critic params to their target counterparts
        params = unfreeze(network_params)
        for g in range(n_goals):
            params[f'modules_target_critic_goal_{g}'] = params[f'modules_critic_goal_{g}']
        params = freeze(params)


        network = TrainState.create(network_def, network_params, tx=network_tx)

        config = dict(**config)
        config['action_dim'] = action_dim
        if config.get('target_entropy') is None:
            # set target entropy w.r.t full action dim
            config['target_entropy'] = -float(full_action_dim) / 2

        return cls(rng, network=network, config=flax.core.FrozenDict(**config), n_goals=n_goals)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='multi_goal_sac',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512),
            value_hidden_dims=(512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            num_qs=2,
            encoder=None,
            horizon_length=1,
            action_chunking=False,
            init_temperature=1.0,
            target_entropy=None,
            critic_loss_type='mse',
            entropy_backup=False,
        )
    )
    return config

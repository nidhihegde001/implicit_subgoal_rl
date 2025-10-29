from collections import defaultdict
import os
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from functools import partial


def supply_rng(f, rng=None):
    """Helper to split RNG before each call to f(rng=...)."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)
    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def returns_to_go_from_indices(rewards, indices, discount=0.99):
    """
    Compute MC returns-to-go from each index in `indices` to the end:
        G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
    """
    T = len(rewards)
    # Precompute discounted suffix sums in O(T)
    G = 0.0
    suffix = np.zeros(T, dtype=np.float32)
    for t in reversed(range(T)):
        G = rewards[t] + discount * G
        suffix[t] = G
    # Also precompute powers of gamma
    pow_gamma = np.ones(T+1, dtype=np.float32)
    for k in range(1, T+1):
        pow_gamma[k] = pow_gamma[k-1] * discount
    # For index t, return is suffix[t] (already correct since it’s from t)
    # If you wanted arbitrary offsets, you'd divide by pow_gamma[offset], but not needed here.
    return [float(suffix[t]) for t in indices]


def visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix, save_dir='./plots'):
    if not scatter_data:
        return

    save_dir = os.path.join(save_dir, 'q_accuracy_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)
    reward_desc = "MC return-to-go (episode end)"
    fig.suptitle(f'Q-Function Accuracy Analysis ({reward_desc})\nStep {global_step} ({suffix})', fontsize=12)

    # Subplot 1: per-episode time series
    ax1 = axes[0]
    for i, episode in enumerate(time_series_data):
        timesteps = np.arange(len(episode['q_preds']))
        color = plt.cm.coolwarm(i / max(1, len(time_series_data) - 1))
        if len(timesteps) == 0:
            continue
        ax1.plot(timesteps, episode['q_preds'], marker='s', linestyle='--',
                 markersize=4, color=color, alpha=0.5, label='Q-Pred' if i == 0 else None)
        ax1.plot(timesteps, episode['mc_returns'], marker='^', linestyle='-',
                 markersize=4, color=color, alpha=0.5, label='MC Return' if i == 0 else None)

    ax1.set_title('Q-Value vs MC Return at Decision Points')
    ax1.set_xlabel('Decision Point Index (per episode)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: scatter correlation
    ax2 = axes[1]
    q_preds = [d[0] for d in scatter_data]
    mc_returns = [d[1] for d in scatter_data]

    ax2.scatter(q_preds, mc_returns, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    if q_preds and mc_returns:
        min_val = min(min(q_preds), min(mc_returns))
        max_val = max(max(q_preds), max(mc_returns))
        ax2.plot([min_val, max_val], [min_val, max_val], '--', alpha=0.75, linewidth=2, label='y = x')

    ax2.set_title('Predicted Q vs MC Return (All Episodes)')
    ax2.set_xlabel('Predicted Q-Value')
    ax2.set_ylabel('MC Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    if len(q_preds) > 1:
        correlation = np.corrcoef(q_preds, mc_returns)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    save_path = os.path.join(save_dir, f'q_accuracy_step_{global_step}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-function accuracy plot saved to: {save_path}")

def evaluate(
    agent,
    env,
    global_step,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    actor_fn=None,
    env_name='',
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        actor_fn: Optional custom actor function. If None, uses agent.sample_actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if actor_fn is None:
        suffix = "SDE"
        actor_fn = agent.sample_actions
    else:
        suffix = "ODE"
   
    actor_fn = supply_rng(actor_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    renders = []
    
    scatter_data = []   # list of (q_pred, mc_return) across all episodes
    time_series_data = []  # per-episode lists
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        obs = env.reset()
        
        done = False
        step = 0
        render = []
        # action_chunk_lens = defaultdict(lambda: 0)
        
        rewards = []                     # per-step rewards
        decision_indices = []            # step indices where a new chunk was decided
        q_preds_at_decisions = []        # Q(s_t, a_t) at each decision point
        # action_chunk_lens = defaultdict(int)

        # action_queue = []
        while not done:
            # if len(action_queue) == 0:
            obs_array = np.concatenate([v.ravel() for v in obs.values()])
            action = actor_fn(obs_array) #, temperature=eval_temperature)
            # print("Action shape: ", action.shape)
            action = np.array(action).reshape(-1, action_dim) # (B, action_dim)
            # action_chunk_len = action.shape[0]
            # for a in action:
            #     action_queue.append(a)
            q_pred_g = []
            for g in agent.n_goals:
                critic_name = f"target_critic_goal_{g}"
                # mean of ensembles
                q_pred = jnp.mean(agent.select(critic_name)(jax.device_put(obs[np.newaxis, ...]), 
                                            jax.device_put(action)))
                q_pred_g.append(float(q_pred))
                
            q_preds_at_decisions.append(q_pred_g) # weight these critic preds with w_is
            decision_indices.append(step)
            # action_chunk_lens[f"action_chunk_length_{action_chunk_len}"] += 1
            # info['action_chunk_length'] = action_chunk_lens
            
            # action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_obs, reward, done, info = env.step(np.clip(action, -1, 1))
            rewards.append(float(reward))
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=obs,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            obs = next_obs

        mc_returns = returns_to_go_from_indices(rewards, decision_indices, discount=0.99)
        time_series_data.append({
            'q_preds': q_preds_at_decisions,
            'mc_returns': mc_returns,
        })
        scatter_data.extend(list(zip(q_preds_at_decisions, mc_returns)))

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix=env_name)

    return stats, trajs, renders

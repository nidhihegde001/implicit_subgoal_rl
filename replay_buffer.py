import numpy as np
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from utils.augmentations import batched_random_crop, color_transform

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Optimized JAX-compatible dataset class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

    
    def _init_rng(self, seed=42):
        return jax.random.PRNGKey(seed)

    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if 'states' not in data:
             data['states'] = data['observations']
             data['next_states'] = data['next_observations']
             
        if freeze:
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    v.setflags(write=False)
        return cls(data)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)

        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs, next_obs = [], []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        
        if self.p_aug is not None and batch['observations'].ndim > 3 and np.random.rand() < self.p_aug:
            self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        B, T = batch_size, sequence_length
        max_start = self.size - T
        idxs = np.random.randint(max_start + 1, size=B)
        next_action_idxs = np.minimum(idxs + T, self.size - T)

        offs = np.arange(T)[None, :]        # (1, T)
        seq_idxs = idxs[:, None] + offs     # (B, T)
        next_seq_idxs = next_action_idxs[:, None] + offs

        # Core sequences
        obs_seq          = self['observations'][seq_idxs]
        next_obs_seq     = self['next_observations'][seq_idxs]
        actions_seq      = self['actions'][seq_idxs]
        next_actions_seq = self['actions'][next_seq_idxs]
        rewards_seq      = self['rewards'][seq_idxs]                   # (B, T)
        subgoal_seq      = self['subgoal_rewards'][seq_idxs]           # (B, T, n_subgoals)
        goal_done_seq    = self['goal_done'][seq_idxs]                 # (B, T, n_subgoals)
        phases_seq       = self['phases'][seq_idxs]                    # (B, T)
        dones_seq        = self['dones'][seq_idxs]                     # (B, T)

        states_seq       = self['states'][seq_idxs]
        next_states_seq  = self['next_states'][seq_idxs]

        # Masks / terminal logic (same logic)
        masks_seq        = 1.0 - dones_seq
        terminals_seq    = dones_seq
        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        valid = np.ones_like(masks_seq)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # Discounted returns-to-go
        disc = (discount ** np.arange(T)).astype(rewards_seq.dtype)
        rewards_prefix = np.cumsum(rewards_seq * disc[None, :], axis=1)

        first_obs = obs_seq[:, 0, ...]
        last_obs  = next_obs_seq[:, 0, ...]

        if self.p_aug is not None and first_obs.ndim > 3 and np.random.rand() < self.p_aug:
            tmp = {'observations': first_obs, 'next_observations': last_obs}
            self.augment(tmp, ['observations', 'next_observations'])
            first_obs, last_obs = tmp['observations'], tmp['next_observations']

        return dict(
            observations       = first_obs,
            next_observations  = last_obs,
            actions            = actions_seq,
            next_actions       = next_actions_seq,
            rewards            = rewards_seq,          # Per step reward
            rewards_prefix     = rewards_prefix,       # Discounted cumulative reward
            subgoal_rewards    = subgoal_seq,          
            goal_done          = goal_done_seq, 
            phases             = phases_seq,         
            dones              = dones_seq,            
            masks              = masks_prefix,
            terminals          = terminals_prefix,
            valid              = valid,
            states             = states_seq[:, 0, ...],
            next_states        = next_states_seq[:, 0, ...],
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result
    
    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(jax.tree_util.tree_leaves(batch[keys[0]])[0])
        rng = self._init_rng()
        rng, key1, key2 = jax.random.split(rng, 3)
        crop_froms_2d = jax.random.randint(key1, (batch_size, 2), 0, 2 * padding + 1)
        zeros = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        crop_froms = jnp.concatenate([crop_froms_2d, zeros], axis=1)

        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )
        if 'observations' in batch:
            batch['observations'] = (color_transform(key2, batch['observations'] / 255.0) * 255.0).astype(np.uint8)



class MultiGoalReplayBuffer(Dataset):
    """
    Dataset-style replay buffer for multi-subgoal RL.
    Uses JAX PyTree (dict of arrays) internally like your original buffer.
    """

    def __init__(self, buffer_dict, n_subgoals):
        super().__init__(**buffer_dict)
        self._dict = buffer_dict  # pytree of arrays
        self.max_size = next(iter(buffer_dict.values())).shape[0]
        self.size = 0
        self.pointer = 0
        self.phase_counts = np.zeros(n_subgoals, dtype=int)

    @classmethod
    def create(cls, state_shape, action_shape, buffer_size, n_subgoals=7):
        """
        Create an empty replay buffer with correct shapes.
        """
        buffer_dict = {
            "states": np.zeros((buffer_size, *state_shape), dtype=np.float32),
            "actions": np.zeros((buffer_size, *action_shape), dtype=np.float32),
            "next_states": np.zeros((buffer_size, *state_shape), dtype=np.float32),
            "rewards": np.zeros((buffer_size,), dtype=np.float32),
            "subgoal_rewards": np.zeros((buffer_size, n_subgoals), dtype=np.float32), # for now, 0 if goal_done, -1 if not
            "goal_done": np.zeros((buffer_size, n_subgoals), dtype=np.float32),
            "phases": np.zeros((buffer_size,), dtype=np.int32), # current phase
            "dones": np.zeros((buffer_size,), dtype=np.float32),
        }
        return cls(buffer_dict, n_subgoals)

    def add_transition(self, state, action, next_state, reward_total,
                       subgoal_reward_vector, goal_done, phase, done):
        """
        Add one transition (same fields as your old buffer).
        """

        self._dict["states"][self.pointer] = state
        self._dict["actions"][self.pointer] = action
        self._dict["next_states"][self.pointer] = next_state
        self._dict["rewards"][self.pointer] = reward_total
        self._dict["subgoal_rewards"][self.pointer] = subgoal_reward_vector
        self._dict["goal_done"][self.pointer] = goal_done
        self._dict["phases"][self.pointer] = phase
        self._dict["dones"][self.pointer] = float(done)

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if phase < len(self.phase_counts):
            self.phase_counts[phase] += 1

    def sample(self, batch_size):
        """
        Sample a batch and return in JAX array form.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            key: jnp.array(value[idxs])
            for key, value in self._dict.items()
        }

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        self.pointer = 0


    def print_phase_distribution(self):
        print("\n===== Replay Buffer Phase Distribution =====")
        print(f"Total transitions stored: {self.size}")
        for p, count in enumerate(self.phase_counts):
            print(f"Phase {p}: {count} transitions")
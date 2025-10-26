import numpy as np
import os, json, random
from absl import app, flags, logging
from tqdm import trange
import robosuite as suite
print("robosuite:", suite.__version__)

from replay_buffer import MultiGoalReplayBuffer
from agents import agent_dict
from agents.sac import get_config

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS


# flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
# flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
# flags.DEFINE_string('project', 'test', 'Project name.')
# flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
# flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('num_init_steps', 100, 'Initial steps to fill RB')
# flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
# flags.DEFINE_integer('save_interval', -1, 'Save interval.')
# flags.DEFINE_integer('start_training', 5000, 'when does training start')

# flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

# flags.DEFINE_float('discount', 0.99, 'discount factor')

# flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
# flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
# flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

# flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
# flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for 1B datasets because of memory constraints')
# flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

# flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
# flags.DEFINE_bool('sparse', False, "make the task sparse reward")
# flags.DEFINE_float('offline_ratio', -1.0, "ratio of offline data to use in mixed training (-1 = current naive scheme, 0.0 = pure online, 1.0 = pure offline)")
# flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')



def main(_):
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")

    # Manually override parameters to match your printed config
    controller_config.update(dict(
        control_delta=True,
        damping=1,
        damping_limits=[0, 10],
        impedance_mode="fixed",
        input_max=1,
        input_min=-1,
        interpolation=None,
        kp=150,
        kp_limits=[0, 300],
        orientation_limits=None,
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        position_limits=None,
        ramp_ratio=0.2,
        type="OSC_POSE",
        uncouple_pos_ori=True,
    ))

    # -------------------------------
    # 2. Create the environment
    # -------------------------------
    env = suite.make(
        env_name="TwoArmTransport",
        robots=["Panda", "Panda"],
        env_configuration="single-arm-opposed",
        controller_configs=controller_config,
        control_freq=20,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=False,  # Set this to False to avoid payload_pos observable
        reward_shaping=True,
        ignore_done=True,
        render_gpu_device_id=0,
    )

    env.reward_mode = "key" # 3 sub-goals
    env.n_goals = 3

    # Confirm setup
    print(f"Created environment with name {env.__class__.__name__}")
    print(f"Action size is {env.action_dim}")
    print("============= Loaded Environment =============")
    print(env)
    print("==============================================")
    # -----------------------------------
    # 3. Run rollout
    # -----------------------------------
    obs = env.reset()

    # Replay Buffer
    obs_array = np.concatenate([v.ravel() for v in obs.values()])
    obs_dim = obs_array.shape[0]  # total dimension
    act_dim = env.action_dim
    # sub-goals
    n_total_subgoals = 7
    active_goals = [1, 3, 6] if env.reward_mode == "key" else list(range(n_total_subgoals))

    print("Num sub-goals: ", len(active_goals))
    buffer_size = FLAGS.buffer_size
    replay_buffer = MultiGoalReplayBuffer.create(
        state_shape=(obs_dim,),
        action_shape=(act_dim,),
        buffer_size=buffer_size,
        n_subgoals=env.n_goals,
    )


    # -------------------------------
    # Hyperparameters
    # -------------------------------
    num_init_steps = FLAGS.num_init_steps       # Fill buffer with random actions first
    total_training_steps = 200_000
    batch_size = 256
    update_after = num_init_steps  # Start gradient updates only after buffer has enough data
    update_every = 1               # Train every environment step (can increase for speed)

    # -------------------------------
    # 1. Random rollout to fill buffer
    # -------------------------------
    obs = env.reset()
    obs_array = np.concatenate([v.ravel() for v in obs.values()])

    log_step = 0
    for step in trange(num_init_steps, desc="Filling Replay Buffer with Random Policy"):
        log_step += 1
        action = np.random.uniform(low=-1, high=1, size=env.action_dim)  # Random action

        next_obs, reward, done, info = env.step(action)
        info = env.reward_dict  # use your reward dictionary from env

        # ----- Flatten obs -----
        obs_array = np.concatenate([v.ravel() for v in obs.values()])
        next_obs_array = np.concatenate([v.ravel() for v in next_obs.values()])

        # ----- Subgoal and phase info -----
        goal_done = np.array(info.get("goal_done", np.zeros(n_total_subgoals)))[active_goals]
        subgoal_reward = np.where(goal_done, 0, -1)
        phase = info.get("phase", 0)

        # ----- Store in Replay Buffer -----
        replay_buffer.add_transition(
            state=obs_array,
            action=action,
            next_state=next_obs_array,
            reward_total=reward,
            subgoal_reward_vector=subgoal_reward,
            goal_done=goal_done,
            phase=phase,
            done=done,
        )

        obs = next_obs if not done else env.reset()

    print(f"Replay buffer filled with {len(replay_buffer)} transitions.")


    # -------------------------------
    # 2. SAC Training Loop
    # -------------------------------

    agent_class = agent_dict['sac']
    agent = agent_class.create(
        FLAGS.seed,
        obs_array,
        action,
        get_config(),
    )
    
    obs = env.reset()
    for step in trange(total_training_steps, desc="SAC Training"):

        # Select action from SAC policy (with optional exploration noise if needed)
        action = agent.sample_actions(obs_array) 

        # Step environment
        next_obs, reward, done, info = env.step(action)
        info = env.reward_dict

        # Flatten observations
        obs_array = np.concatenate([v.ravel() for v in obs.values()])
        next_obs_array = np.concatenate([v.ravel() for v in next_obs.values()])

        # Subgoal + phase
        goal_done = np.array(info.get("goal_done", np.zeros(n_total_subgoals)))[active_goals]
        subgoal_reward = np.where(goal_done, 0, -1)
        phase = info.get("phase", 0)

        # Add new transition to replay buffer (even during training!)
        replay_buffer.add_transition(
            state=obs_array,
            action=action,
            next_state=next_obs_array,
            reward_total=reward,
            subgoal_reward_vector=subgoal_reward,
            goal_done=goal_done,
            phase=phase,
            done=done,
        )

        # After enough data is collected, start updating SAC
        if step >= update_after and step % update_every == 0:
            batch = replay_buffer.sample(batch_size)

            # Perform 1 gradient update on critic + actor
            agent, loss_info = agent.update(batch)
            if step % FLAGS.log_interval == 0:
                # logger.log(loss_info, "agent info", step=log_step)
                print(loss_info)

        # Reset environment if terminated
        obs = next_obs if not done else env.reset()

    env.close()
    print("Training complete ✅")



if __name__ == '__main__':
    app.run(main)  # This parses FLAGS
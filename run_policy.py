import robosuite as suite
import numpy as np
from replay_buffer import MultiGoalReplayBuffer
print(suite.__version__)


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

buffer_size = int(1e6)
replay_buffer = MultiGoalReplayBuffer.create(
    state_shape=(obs_dim,),
    action_shape=(act_dim,),
    buffer_size=buffer_size,
    n_subgoals=env.n_goals,
)


num_init_steps = 10
obs = env.reset()
num_episodes = 0
for step in range(num_init_steps):
    action = np.random.uniform(low=-1, high=1, size=env.action_dim)
    next_obs, reward, done, info = env.step(action)
    info = env.reward_dict

    # flatten obs and next_obs
    obs_array = np.concatenate([v.ravel() for v in obs.values()])
    next_obs_array = np.concatenate([v.ravel() for v in next_obs.values()])


    # Construct subgoal-related vectors
    goal_done = np.array(info.get("goal_done", np.zeros(n_total_subgoals)))[active_goals]
    subgoal_reward = np.where(goal_done, 0, -1)
    phase = info.get("phase", 0)

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

    if (step + 1) % 10 == 0:
        print(f"\n--- Step {step+1} Buffer Diagnostics ---")
        replay_buffer.print_phase_distribution()
    
    if done:
        num_episodes += 1


    obs = next_obs if not done else env.reset()

env.close()

print("Num episodes traversed: ", num_episodes)

# # --------------------------
# # 4. Verify Buffer Contents
# # --------------------------

print("\n===== Buffer Diagnostics =====")
print(f"Buffer current size    : {replay_buffer.size}")
print(f"Buffer expected <= size: {num_init_steps}")
print(f"Pointer position       : {replay_buffer.pointer}")

# --------------------------
# 5. Sample from buffer
# --------------------------
batch = replay_buffer.sample(batch_size=4)

print("\n===== Sampled Batch =====")
print("Batch states shape        :", batch["states"].shape)
print("Batch actions shape       :", batch["actions"].shape)
print("Batch next states shape   :", batch["next_states"].shape)
print("Batch subgoal rewards     :", batch["subgoal_rewards"])
print("Batch goal done flags     :", batch["goal_done"])
print("Batch phases              :", batch["phases"])
print("Batch dones               :", batch["dones"])

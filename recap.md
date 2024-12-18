### How TD3 (Twin Delayed Deep Deterministic Policy Gradient) Works

**TD3** is specifically designed to solve problems in **continuous action spaces**, such as our custom version of the Lunar Lander. This algorithm was developed from DDPG (Deep Deterministic Policy Gradient) and addresses issues like **overestimation bias** and instability during training.

The TD3 agent interacts with the environment to learn an **optimal policy** that maximizes cumulative rewards over time. To achieve this, it uses two main types of Neural Networks:

1. **Actor Network (Policy)**: Determines the best action to take in a given state.
2. **Critic Networks**: Estimate the quality (or Q-value) of the actions suggested by the policy.

#### Key Components of TD3

- **Policy**: The policy maps states to actions deterministically. Given a state $ s $, the actor outputs an action $ a = \mu(s; \theta_\mu) $, where $ \theta_\mu $ are the actor network's parameters.

- **Q-Values**: Q-values measure the expected return when taking an action $ a $ in state $ s $ and following the policy thereafter. Critic networks estimate these Q-values, guiding the actor network to improve the policy.

#### TD3 Features and Improvements

**Twin Critics**:
   TD3 uses **two independent critic networks** to estimate Q-values ($ Q_1 $ and $ Q_2 $). To reduce **overestimation bias**, it uses the minimum value between the two critics during updates:
   $$
   Q_\text{target}(s, a) = r + \gamma \cdot \min(Q_1(s', a'), Q_2(s', a'))
   $$
   Where:
   - $ r $: reward received for taking action $ a $ in state $ s $.
   - $ \gamma $: discount factor.
   - $ s' $: next state.
   - $ a' $: action chosen by the actor in the next state.

**Delayed Policy Updates**:
   The policy (actor) and **target networks** are updated **less frequently** than the critic networks. Typically, they are updated every two critic updates, reducing variance and increasing training stability.

**Target Networks**:
   Target networks are delayed versions of the main networks (actor and critics). They are updated gradually using **soft updates**:
   $$
   \theta_\text{target} \gets \tau \theta_\text{main} + (1 - \tau) \theta_\text{target}
   $$
   Where $ \tau $ (e.g., 0.005) is the update factor.

**Clipped Double Q-Learning**:
   TD3 uses the minimum value of the two critics to avoid overly optimistic updates:
   $$
   y = r + \gamma \cdot \min(Q_1(s', a'), Q_2(s', a'))
   $$

**Target Policy Smoothing**:
   During Q-value updates, TD3 adds **noise to the target action** to make the policy more robust to small changes:
   $$
   a' = \text{clip}(\mu(s'; \theta_\text{target}) + \epsilon, \text{low}, \text{high})
   $$
   Where $ \epsilon \sim \mathcal{N}(0, \sigma) $ is a small Gaussian noise.

---

### TensorBoard Metrics

The following metrics help monitor TD3 training:

- **eval/mean_ep_length**: Average episode length during evaluation (how many timesteps episodes last).
- **eval/mean_reward**: Average reward obtained during evaluation.
- **rollout/ep_len_mean**: Average episode length during training rollouts.
- **rollout/ep_rew_mean**: Average reward obtained during training rollouts.
- **time/fps**: Frames per second during training (higher values indicate more efficient training).
- **train/actor_loss**: Indicates how much the policy network is being adjusted during training.
- **train/critic_loss**: Measures the error in Q-value predictions by the critic network.
- **train/learning_rate**: Current learning rate of the model.

## Optuna
Optuna is used to systematically search for the best hyperparameters for the TD3 algorithm.
The following hyperparameters are optimized:
- Learning Rate: Controls the speed of updates to the network weights.
- Batch Size: Number of samples processed in each training step.
- Tau: Soft update factor for the target networks.
- Gamma: Discount factor for future rewards.
- Buffer Size: Size of the replay buffer, which stores past experiences for training.

```python
learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
tau = trial.suggest_float("tau", 0.005, 0.05)
gamma = trial.suggest_float("gamma", 0.95, 0.999)
buffer_size = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000])
```

## Changes to the Environment
The custom Lunar Lander environment introduces several changes to make the task more challenging and flexible. It allows customization of parameters like gravity, wind, and turbulence, as well as adding observation noise to simulate sensor inaccuracies. The environment supports partial observability by removing specific observation elements, such as angular velocity and left leg contact, making it harder for the agent. The reward system is modified to penalize excessive engine usage, unstable angles, and prolonged time to land, encouraging efficient and stable landings. Observations are clipped to remain within valid bounds, and the environment provides options for more dynamic experimentation. These changes make the task more realistic and suitable for testing robust reinforcement learning algorithms.
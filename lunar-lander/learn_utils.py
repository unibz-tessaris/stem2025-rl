import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import gymnasium as gym

from typing import Any, Iterable, Sequence, Tuple, Union


class Discretizer:
    def __init__(self, env, num_bins=(6, 6, 6, 6, 6, 6, 2, 2)):
        """Initialize the Discretizer with environment and bin settings."""
        self.num_bins = num_bins
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

        # Override specific state bounds
        self.state_bounds[0] = (-1.0, 1.0)
        self.state_bounds[1] = (-0.2, 1.43)
        self.state_bounds[6] = (0.01, 1.0)  # Adjust leg contact
        self.state_bounds[7] = (0.01, 1.0)

    def discretize_uniform(self, state):
        """Convert continuous state into a discrete index."""
        indices = []
        for i, s in enumerate(state):
            lb, ub = self.state_bounds[i]
            bins = np.linspace(lb, ub, self.num_bins[i] - 1)
            indices.append(np.digitize(s, bins))
        return tuple(indices)

    def discretize(self,state):
        """Convert continuous state into a discrete index. Bins are created in a non-uniform way,
        favoring the median value"""
        indices = []
        for i, s in enumerate(state):
            lb, ub = self.state_bounds[i]
            if i < 6:
                segment = (ub - lb) * 0.15 # 0.15
                bins = np.array([0 - segment, 0 - segment / 2, 0, 0 + segment / 2, 0 + segment])
                if( i == 1 ):  # Y coordinate requires ad-hoc treatment
                    segment = (ub - lb) * 0.25 # 0.25
                    bins = np.array([segment/2, segment, segment+segment/2, 2*segment])
                indices.append(np.digitize(s, bins))
            else:
                bins = np.linspace(lb, ub, 1)
                indices.append(np.digitize(s, bins))
        return tuple(indices)

def plot_rewards(rewards, ax = None):
    """Creates a plot showing the training history of the given list of rewards.
    Returns the axes in which the plot has been created.
    """
    # rews = np.array(rewards).T
    # # calculate the running average <https://stackoverflow.com/a/30141358>
    # smoothed_rews = pd.Series(rews).rolling(max(1, int(len(rews) * .01))).mean()
    rews = np.array(rewards)
    window_size = max(1, int(len(rews) * 0.01))  # Define the rolling window size

    # Compute the rolling mean using NumPy's convolution
    smoothed_rews = np.convolve(rews, np.ones(window_size) / window_size, mode='valid')

    ax = ax if ax is not None else plt.axes()

    ax.plot(smoothed_rews)
    ax.plot([np.mean(rews)] * len(rews), label='Mean', linestyle='--')
    ax.plot(rews, color='grey', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_ylim(None, 1500)

    return ax


def test(discretizer, Q):
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
               wind_power=0, turbulence_power=0)

    rewards = []
    for episode in range(100):
        # Let's deploy
        state, _ = env.reset(seed=42)
        state = discretizer.discretize(state)
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = discretizer.discretize(next_state)
            total_reward += reward

            state = next_state
        rewards.append(total_reward)
    print('Average deployment reward: {}'.format(np.mean((rewards))))


# It takes a QTable and displays a run of the lander
def view(discretizer, Q):

    env = gym.make("LunarLander-v3", render_mode="human", continuous=False, gravity=-10.0, enable_wind=False,
               wind_power=0, turbulence_power=0)

    # Let's deploy
    state, _ = env.reset(seed=42)
    state = discretizer.discretize(state)
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = discretizer.discretize(next_state)
        total_reward += reward

        state = next_state

    env.close()

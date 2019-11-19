import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 10000
GAMMA = 0.9

log_interval = 1  # log training metrics every log_interval steps (default everystep)


class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, lr=3e-4):
        super(Policy, self).__init__()
        self.lr = lr
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """
        forward of the policy network.
        :param state: observation state made on the environment, (1, state_space) tensor.
        :return: actions distribution (softmax)
        """
        x = self.linear1(state)
        x = self.linear2(x)
        return F.softmax(x, dim=1)

    def get_action(self, state):
        """
        use the forward function to get actions distribution, then choose action based on returned probabilities
        :param state: observation state made on the environment, a numpy array of size (state_space)
        :return: an action to take
        """
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, state_space)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        # we store the log_probability for each step in order to compute policy gradient
        log_prob = m.log_prob(action)

        return action.item(), log_prob


def update_policy(policy_network, rewards, log_probs):
    """
    Function that update policy network after each episode based on policy gradient function.
    :param policy_network:
    :param rewards: list of rewards obtain along the episode
    :param log_probs: output log_probs of policy for each action taken along the episode
    """
    discounted_rewards = list()
    # for every step, we calculate the discounted reward
    for t in range(len(rewards)):  # or nbr of steps
        Gt = 0
        pw = 0
        # discounted reward : starting in a state, what is the sum of expected rewards we can expect ?
        # Of course, we compute it retrospectively.
        for r in rewards[t:]:
            Gt += GAMMA**pw * r
            pw += 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    # Normalisation of discounted rewards help for training stability, it helps to reduce the policy variance in a way.
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = list()
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.cat(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    policy = Policy(env.observation_space.shape[0], env.action_space.n, 128)
    step_per_episode = list()
    avg_step_per_episode = list()
    all_rewards = list()
    running_reward = 10

    for e in range(MAX_EPISODES):
        state = env.reset()
        log_probs = list()
        rewards = list()

        for step in range(MAX_STEPS_PER_EPISODE):
            env.render()
            action, log_prob = policy.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy, rewards, log_probs)
                step_per_episode.append(step)
                # we compute the moving average of number of episodes for the 10 last episodes
                avg_step_per_episode.append(np.mean(step_per_episode[-10:]))
                all_rewards.append(np.sum(rewards))

                # print the results only after n episodes
                if e % log_interval == 0:
                    print('Episode {}, Total reward: {:.2f}, Avg reward: {:2f}, len(e): {}'.format(
                        e,
                        np.sum(rewards),
                        np.mean(all_rewards[-10:]),
                        step))
                break

            state = new_state

        running_reward = 0.05 * np.sum(rewards) + (1 - 0.05) * running_reward
        if running_reward > env.spec.reward_threshold:
            print('SOLVED ! Running reward is now {} and the last episode runs to {} time steps.'.format(
                running_reward, step))
            break


if __name__ == '__main__':
    main()











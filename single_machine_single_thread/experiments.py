import numpy as np
import gym
import torch
import random

from network import Policy, Critic
from learner import Learner
from actor import Actor
from replay_buffer import ReplayBuffer
from model_server import ModelServer

if __name__ == '__main__':
    num_actors = 4
    num_epochs = 2000
    env_name = 'CartPole-v1'

    variants = {}
    variants['env_name'] = env_name

    test_env = gym.make(variants['env_name'])
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.n

    variants['state_size'] = state_size
    variants['action_size'] = action_size
    variants['hidden_size'] = 128
    variants['traj_len'] = 100
    variants['critic_learning_rate'] = 1e-4
    variants['policy_learning_rate'] = 1e-4
    variants['batch_size'] = 2
    variants['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    variants['seed'] = 1
    random.seed(variants['seed'])
    np.random.seed(variants['seed'])
    torch.manual_seed(variants['seed'])
    if variants['device']=='cuda':
        torch.cuda.manual_seed_all(variants['seed'])

    critic = Critic(state_size,1,variants['hidden_size'])
    policy = Policy(state_size,action_size,variants['hidden_size'])

    replay_buffer = ReplayBuffer(variants)
    model_server = ModelServer(variants)
    learner = Learner(variants,critic,policy,replay_buffer,model_server,test_env)
    actors = [Actor(variants,replay_buffer,model_server) for _ in range(num_actors)]
    for epoch in range(num_epochs):
        for actor in actors:
            actor.collect()
        learner.learn(epoch)
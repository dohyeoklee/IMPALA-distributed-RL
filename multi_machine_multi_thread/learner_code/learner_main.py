import numpy as np
import gym
import torch
import random
import argparse

from learner import Learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--traj_len', default=20, type=int)
    parser.add_argument('--critic_learning_rate', default=1e-3, type=float)
    parser.add_argument('--policy_learning_rate', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--eval_num_rollout', default=5, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_actors', default=4, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    args = parser.parse_args()
    variants = vars(args)
    
    variants['env_name'] = 'CartPole-v1'

    test_env = gym.make(variants['env_name'])
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.n

    variants['state_size'] = state_size
    variants['action_size'] = action_size
    variants['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(variants['seed'])
    np.random.seed(variants['seed'])
    torch.manual_seed(variants['seed'])
    if variants['device']=='cuda':
        torch.cuda.manual_seed_all(variants['seed'])
    
    learner = Learner(variants,test_env)
    
    for epoch in range(variants['num_epochs']):
        learner.learn(epoch)
        learner.evaluate(epoch)
    
    learner.close()
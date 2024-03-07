import ray
import torch
import gym
import time
import os

from network import Policy

@ray.remote
class Actor():
    def __init__(self,actor_id,variants,replay_buffer):
        self.actor_id = actor_id
        self.env = gym.make(variants['env_name'])
        self.state = self.env.reset()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.policy = Policy(self.state_size,self.action_size,variants['hidden_size'])
        self.traj_len = variants['traj_len']
        self.replay_buffer = replay_buffer

    def load_weights(self):
        if 'model.pt' in os.listdir('./model'):
            self.policy = torch.load('model/model_'+str(self.actor_id)+'.pt')

    def collect(self):
        while True:
            self.load_weights()
            traj = {'state':[],'action':[],'reward':[],'next_state':[],'done':[],'action_log_prob':[]}
            for _ in range(self.traj_len):
                dist = self.policy(torch.Tensor(self.state))
                action = dist.sample()
                next_state,reward,done,_ = self.env.step(action.item())
                action_log_prob = dist.log_prob(action).unsqueeze(0).detach().tolist()
                traj['state'].append(list(self.state))
                traj['action'].append(action.item())
                traj['reward'].append(reward)
                traj['next_state'].append(list(next_state))
                traj['done'].append(done)
                traj['action_log_prob'].append(action_log_prob)
                self.state = self.env.reset() if done else next_state
            self.replay_buffer.push.remote(traj)
            time.sleep(0.1)
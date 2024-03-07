import torch
from collections import namedtuple
import gym
from network import Policy

Transition = namedtuple('Transition',['state','action','reward','next_state','done','action_log_prob'])

class Actor():
    def __init__(self,variants,replay_buffer,model_server):
        self.env = gym.make(variants['env_name'])
        self.state = self.env.reset()
        self.seed = variants['seed']
        self.state_size = variants['state_size']
        self.action_size = variants['action_size']
        self.replay_buffer = replay_buffer
        self.model_server = model_server        
        self.policy = Policy(self.state_size,self.action_size,variants['hidden_size'])
        self.traj_len = variants['traj_len']

    def collect(self):
        self.model_server.load_model_weights(self.policy)
        traj = []
        for _ in range(self.traj_len):
            dist = self.policy(torch.Tensor(self.state))
            action = dist.sample()
            next_state,reward,done,_ = self.env.step(action.item())
            action_log_prob = dist.log_prob(action).unsqueeze(0).detach().numpy()
            traj.append(Transition(self.state,action,reward,next_state,done,action_log_prob))
            if done: self.state = self.env.reset()
            else: self.state = next_state
        self.replay_buffer.push(traj)
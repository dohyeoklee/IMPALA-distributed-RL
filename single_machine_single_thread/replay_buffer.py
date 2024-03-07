import numpy as np
from collections import deque
import torch

class ReplayBuffer():
    def __init__(self,variants):
        self.buffer_size = 1000
        self.buffer = deque(maxlen=self.buffer_size)
        self.traj_size = variants['traj_len']
        self.batch_size = variants['batch_size']
        self.device = variants['device']      

    def push(self,traj):
        self.buffer.append(traj)

    def get_batch(self):
        if len(self.buffer)>=self.batch_size:
            idxs = np.random.choice(len(self.buffer),self.batch_size,replace=False)
            state,action,reward,next_state,done,old_action_log_prob = [],[],[],[],[],[]
            for idx in idxs:
                state.append(torch.tensor(np.array([t.state for t in self.buffer[idx]])))
                action.append(torch.tensor([t.action for t in self.buffer[idx]]))
                reward.append(torch.tensor([[t.reward] for t in self.buffer[idx]]))
                next_state.append(torch.tensor(np.array([t.next_state for t in self.buffer[idx]])))
                done.append(torch.tensor(np.array([[1-t.done] for t in self.buffer[idx]])))
                old_action_log_prob.append(torch.tensor(np.array([t.action_log_prob for t in self.buffer[idx]])))
            state = torch.stack(state,dim=0).to(self.device)
            action = torch.stack(action,dim=0).to(self.device)
            reward = torch.stack(reward,dim=0).to(self.device)
            next_state = torch.stack(next_state,dim=0).to(self.device)
            done = torch.stack(done,dim=0).to(self.device)
            old_action_log_prob = torch.stack(old_action_log_prob,dim=0).to(self.device)
            return [state,action,reward,next_state,done,old_action_log_prob]
        else:
            return []
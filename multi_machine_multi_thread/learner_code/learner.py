import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import json
import redis
import socket
from torch.utils.tensorboard import SummaryWriter

from network import Policy, Critic

class Learner():
    def __init__(self,variants,test_env):
        self.variants = variants
        self.writer = SummaryWriter()
        self.state_size = variants['state_size']
        self.action_size = variants['action_size']
        self.critic = Critic(variants['state_size'],1,variants['hidden_size']).to(variants['device'])
        self.policy = Policy(variants['state_size'],variants['action_size'],variants['hidden_size']).to(variants['device'])
        self.redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.batch_size = variants['batch_size']
        self.test_env = test_env
        self.loss = nn.MSELoss()
        self.device = variants['device']
        self.critic_optimizer = Adam(self.critic.parameters(),lr=variants['critic_learning_rate'],eps=1e-1)
        self.policy_optimizer = Adam(self.policy.parameters(),lr=variants['policy_learning_rate'],eps=1e-1)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9995)
        self.policy_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.9995)
        self.ip_ports = ("192.168.0.3",1991)
        self.py_socket = socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)
        self.py_socket.connect(self.ip_ports)

    def close(self):
        self.py_socket.close()

    def evaluate(self,epoch):
        reward_sum = 0
        for _ in range(self.variants['eval_num_rollout']):
            state = self.test_env.reset()
            for _ in range(self.test_env.spec.max_episode_steps):
                state = torch.Tensor(state).to(self.device)
                dist = self.policy(state)
                action = dist.sample()
                next_state,reward,done,_ = self.test_env.step(action.item())
                reward_sum+=reward
                if done: break
                state = next_state
        self.writer.add_scalar("Eval/return",reward_sum/self.variants['eval_num_rollout'],epoch)

    def get_batch(self):
        data_len = min(self.redis_server.llen('impala_data'),self.variants['buffer_size'])
        if data_len>=self.batch_size:
            idxs = np.random.choice(data_len,self.batch_size,replace=False)
            state,action,reward,next_state,done,old_action_log_prob = [],[],[],[],[],[]
            for idx in idxs:
                traj = self.redis_server.lrange('impala_data',int(idx),int(idx))[0]
                traj = json.loads(traj)
                state.append(torch.tensor(np.array(traj['state']),dtype=torch.float32))
                action.append(torch.tensor(traj['action']))
                reward.append(torch.tensor(traj['reward'],dtype=torch.float32))
                next_state.append(torch.tensor(np.array(traj['next_state']),dtype=torch.float32))
                done.append(torch.tensor(np.array(traj['done'])))
                old_action_log_prob.append(torch.tensor(np.array(traj['action_log_prob']),dtype=torch.float32))
            state = torch.stack(state,dim=0).to(self.device)
            action = torch.stack(action,dim=0).unsqueeze(-1).to(self.device)
            reward = torch.stack(reward,dim=0).unsqueeze(-1).to(self.device)
            next_state = torch.stack(next_state,dim=0).to(self.device)
            done = torch.stack(done,dim=0).unsqueeze(-1).to(self.device)
            old_action_log_prob = torch.stack(old_action_log_prob,dim=0).to(self.device)
            return [state,action,reward,next_state,done,old_action_log_prob]
        else:
            return []
        
    def send_weights(self):
        send_data = {k: v.cpu().tolist() for k,v in self.policy.state_dict().items()}
        body = json.dumps(send_data)
        print(len(body))
        msg = bytes(body,'utf-8')
        self.py_socket.sendall(msg)

    def learn(self,epoch):
        #1. get batch
        batching_time_start = time.time()
        batch = self.get_batch()
        if len(batch)>0:
            batching_time_end = time.time()
            #2. compute loss
            forward_time_start = time.time()
            state,old_action,reward,next_state,done,old_action_log_prob = batch
            dist = self.policy(state)
            action_log_prob = dist.log_prob(old_action.squeeze()).unsqueeze(-1)
            critic_value = self.critic(state)
            next_critic_value = self.critic(next_state)            
            discounted = (~done)*0.99            
            
            importance_weight = torch.exp(action_log_prob-old_action_log_prob).detach()
            c_bar,rho_bar = 1,1
            c,rho = torch.clamp(importance_weight,max=c_bar),torch.clamp(importance_weight,max=rho_bar)
            delta_v = rho*(reward+discounted*next_critic_value-critic_value)

            v_trace = [critic_value[:,-1]+delta_v[:,-1]]
            for t in range(delta_v.shape[1]-2,-1,-1):
                v_t = critic_value[:,t]+delta_v[:,t]+0.99*c[:,t]*(v_trace[-1]-critic_value[:,t+1])
                v_trace.append(v_t)
            v_trace = torch.stack([*reversed(v_trace)],dim=1).detach()
            
            advantage = rho*(reward+discounted*v_trace-critic_value).detach()
            entropy_loss = dist.entropy().mean()
            actor_loss = -(action_log_prob*advantage).mean()-0.01*entropy_loss
            critic_loss = self.loss(critic_value,v_trace)

            forward_time_end = time.time()
            #3. weight update
            backward_time_start = time.time()
            self.critic_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.policy_optimizer.step()
            self.critic_lr_scheduler.step()
            self.policy_lr_scheduler.step()
            backward_time_end = time.time()

            self.writer.add_scalar("Train/critic loss",critic_loss.item(),epoch)
            self.writer.add_scalar("Train/avg batch entropy",entropy_loss.item(),epoch)
            self.writer.add_scalar('Train/max batch importance sampling ratio',importance_weight.max().item(),epoch)
            self.writer.add_scalar('Train/avg batch importance sampling ratio',importance_weight.mean().item(),epoch)
            self.writer.add_scalar('Train/min batch importance sampling ratio',importance_weight.min().item(),epoch)
            self.writer.add_scalar('Train/learner batching time',batching_time_end-batching_time_start,epoch)
            self.writer.add_scalar('Train/learner forward time',forward_time_end-forward_time_start,epoch)
            self.writer.add_scalar('Train/learner backward time',backward_time_end-backward_time_start,epoch)
            self.writer.flush()
            
            #4. model weight send to actor
            self.send_weights()
            time.sleep(0.5)
        else:
            time.sleep(0.5)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
from torch.utils.tensorboard import SummaryWriter

class Learner():
    def __init__(self,variants,critic,policy,replay_buffer,model_server,test_env):
        self.writer = SummaryWriter()
        self.state_size = variants['state_size']
        self.action_size = variants['action_size']
        self.critic = critic.to(variants['device'])
        self.policy = policy.to(variants['device'])
        self.replay_buffer = replay_buffer
        self.model_server = model_server
        self.test_env = test_env
        self.loss = nn.MSELoss()
        self.device = variants['device']
        self.critic_optimizer = Adam(self.critic.parameters(),lr=variants['critic_learning_rate'])
        self.policy_optimizer = Adam(self.policy.parameters(),lr=variants['policy_learning_rate'])

    def evaluate(self,epoch):
        reward_sum = 0
        state = self.test_env.reset()
        while True:
            state = torch.Tensor(state).to(self.device)
            dist = self.policy(state)
            action = dist.sample()
            next_state,reward,done,_ = self.test_env.step(action.item())
            reward_sum+=reward
            if done: break
            state = next_state
        print(epoch,reward_sum)
        self.writer.add_scalar("Eval/return",reward_sum,epoch)

    def batch_to_tensor(self,batch):
        state,action,reward,next_state,done,old_action_log_prob = [],[],[],[],[],[]
        for b in batch:
            state.append(torch.tensor(np.array([t[0] for t in b])))
            action.append(torch.tensor([t[1] for t in b]))
            reward.append(torch.tensor([[t[2]] for t in b]))
            next_state.append(torch.tensor(np.array([t[3] for t in b])))
            done.append(torch.tensor(np.array([[1-t[4]] for t in b])))
            old_action_log_prob.append(torch.tensor(np.array([t[5] for t in b])))
        state = torch.stack(state,dim=0).to(self.device)
        action = torch.stack(action,dim=0).to(self.device)
        reward = torch.stack(reward,dim=0).to(self.device)
        next_state = torch.stack(next_state,dim=0).to(self.device)
        done = torch.stack(done,dim=0).to(self.device)
        old_action_log_prob = torch.stack(old_action_log_prob,dim=0).to(self.device)
        return state,action,reward,next_state,done,old_action_log_prob

    def learn(self,epoch):
        #1. batch 받아오기
        batching_time_start = time.time()        
        batch = self.replay_buffer.get_batch()
        if len(batch)>0:
            batching_time_end = time.time()
            #2. compute loss
            forward_time_start = time.time()
            state,old_action,reward,next_state,done,old_action_log_prob = batch
            dist = self.policy(state)
            action_log_prob = dist.log_prob(old_action.squeeze()).unsqueeze(-1)
            critic_value = self.critic(state)
            next_critic_value = self.critic(next_state)            
            discounted = done*0.99            

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
            backward_time_end = time.time()

            self.writer.add_scalar("Train/critic loss",critic_loss.item(),epoch)
            self.writer.add_scalar("Train/avg batch entropy",entropy_loss.item(),epoch)
            self.writer.add_scalar('Train/max batch importance sampling ratio',importance_weight.max().item(),epoch)
            self.writer.add_scalar('Train/avg batch importance sampling ratio',importance_weight.mean().item(),epoch)
            self.writer.add_scalar('Train/min batch importance sampling ratio',importance_weight.min().item(),epoch)
            self.writer.add_scalar('Train/learner batching time',batching_time_end-batching_time_start,epoch)
            self.writer.add_scalar('Train/learner forward time',forward_time_end-forward_time_start,epoch)
            self.writer.add_scalar('Train/learner backward time',backward_time_end-backward_time_start,epoch)

            #5. evaluate
            self.evaluate(epoch)
            self.writer.flush()

            #5. model save
            self.model_server.set_model_weights({k: v.cpu() for k,v in self.policy.state_dict().items()})
import ray
import numpy as np
import torch
import random
import argparse
import time

from actor import Actor
from replay_buffer import ReplayBuffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--traj_len', default=20, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--traj_port', default=1214, type=int)
    parser.add_argument('--model_port', default=1991, type=int)
    parser.add_argument('--num_actors', default=4, type=int)
    parser.add_argument('--server_ip', default='192.168.0.6', type=str)
    args = parser.parse_args()
    variants = vars(args)
   
    variants['env_name'] = 'CartPole-v1'
    variants['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(variants['seed'])
    np.random.seed(variants['seed'])
    torch.manual_seed(variants['seed'])
    if variants['device']=='cuda':
        torch.cuda.manual_seed_all(variants['seed'])

    ray.init()

    replay_buffer = ReplayBuffer.remote(variants)
    actors = [Actor.remote(actor_id,variants,replay_buffer) for actor_id in range(variants['num_actors'])]
    [actor.collect.remote() for actor in actors]
    start_time = time.time()
    while time.time()-start_time<6000:
        ray.wait([replay_buffer.send.remote()])
        time.sleep(0.2)

    ray.get(replay_buffer.close.remote())
    
    ray.shutdown()
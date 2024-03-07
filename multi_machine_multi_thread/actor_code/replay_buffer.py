import ray
import socket
import json
from collections import deque

@ray.remote
class ReplayBuffer():
    def __init__(self,variants):
        self.ip_ports = (variants['server_ip'],variants['traj_port'])
        self.py_socket = socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)
        self.py_socket.connect(self.ip_ports)
        self.buffer = deque(maxlen=10000)

    def send(self):
        body = json.dumps(list(self.buffer))
        msg = bytes(body,'utf-8')
        self.py_socket.sendall(msg)
        self.buffer = deque(maxlen=10000)

    def push(self,traj):
        self.buffer.append(traj)
        
    def close(self):
        self.py_socket.close()
import socket
import json
import torch
import time

from network import Policy

if __name__=='__main__':
    state_size = 4
    action_size = 2
    hidden_size = 32
    num_actors = 4

    local_ip_port = ("192.168.0.3",1991)
    buffer_size = 65000
    ser_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    ser_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ser_socket.bind(local_ip_port)
    ser_socket.listen(1)
    try:
        cli_socket, addr = ser_socket.accept()
        print('connect')
        while True:
            data = cli_socket.recv(buffer_size)
            if len(data)>0:
                my_dict = json.loads(data.decode('utf-8'))
                model_weight = {k: torch.Tensor(v) for k,v in my_dict.items()}
                model = Policy(state_size,action_size,hidden_size)
                for name,param in model.named_parameters():
                    param.data = torch.Tensor(model_weight[name])
                for i in range(num_actors):
                    torch.save(model,'model/model_'+str(i)+'.pt')
                time.sleep(0.1)
    except KeyboardInterrupt:
        ser_socket.close()
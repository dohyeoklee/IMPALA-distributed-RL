import socket
import json
import time
import json
import redis

if __name__=='__main__':
    try:
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        list_key = 'impala_data'
        local_ip_port = ("192.168.0.6",1214)
        buffer_size = 65000
        ser_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        ser_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ser_socket.bind(local_ip_port)
        ser_socket.listen(1)
        cli_socket, addr = ser_socket.accept()
        print('connect')
        start_time = time.time()
        while time.time()-start_time<6000:
            data = cli_socket.recv(buffer_size)
            if len(data)>0:
                data = json.loads(data.decode('utf-8'))
                for d in data:
                    r.lpush(list_key, json.dumps(d))
                    if r.llen('impala_data')>10000:
                        r.lpop(list_key)                                      
            time.sleep(0.1)
        ser_socket.close()
        cli_socket.close()
    except KeyboardInterrupt:
        pass
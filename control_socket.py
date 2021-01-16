import socket
import state

SOCKET_MSG_SLOTH = {
	'start': 0,
	'stop': 1,
	'select': 2
};

def start_control_socket ():
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind(('0.0.0.0', 1236))

	while True:
		data, address = sock.recvfrom(4096)
		print(f'Received {len(data)} bytes from {address}')
		msgs = list(data)

		msg_type = msgs[0]
		if (msg_type == SOCKET_MSG_SLOTH['start']):
			state.enable_visualization()
		elif (msg_type == SOCKET_MSG_SLOTH['stop']):
			state.disable_visualization()
		elif (msg_type == SOCKET_MSG_SLOTH['select']):
			visualization_names = state.get_visualization_names()
			state.set_visualization(visualization_names[msgs[1]])

from __future__ import print_function
from __future__ import division
import socket
import numpy as np
import config

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_pixels(pixels):
    m = np.ndarray([config.N_PIXELS, 4], np.uint8);
    for i in range(config.N_PIXELS):
        r, g, b, w = pixels[0][i], pixels[1][i], pixels[2][i], pixels[3][i]
        m[i][0] = r
        m[i][1] = g
        m[i][2] = b
        m[i][3] = w

    udp_socket.sendto(bytes(m.flatten()), (config.UDP_IP, config.UDP_PORT));

def update(output):
    pixels = output[2].astype(int)
    return send_pixels(pixels)
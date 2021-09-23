from __future__ import print_function
from __future__ import division
# import socket
import numpy as np
import config
from rpi_ws281x import Adafruit_NeoPixel, Color

LED_COUNT = 40        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False

strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
strip.begin()

# udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def update(output):
    pixels = output[2]
    for i in range(config.N_PIXELS):
        r, g, b = pixels[0][i], pixels[1][i], pixels[2][i]

        r = r if r > 1 else 1
        b = b if b > 1 else 1
        g = g if g > 1 else 1
        #if ( not (r and b and b)):
        #   pass
        strip.setPixelColorRGB(i, int(r), int(g), int(b))

    strip.show()

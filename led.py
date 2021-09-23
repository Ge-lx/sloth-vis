import numpy as np
from state import default_config as config
from rpi_ws281x import Adafruit_NeoPixel, Color

LED_COUNT = config['N_PIXELS']
LED_PIN = config['LED_PIN']
LED_FREQ_HZ = config['LED_FREQ_HZ']
LED_BRIGHTNESS = config['LED_BRIGHTNESS']
LED_DMA = 10 # Don't fuck with this. Look it up
LED_INVERT = False

strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
strip.begin()

def update(output):
    pixels = output[2]
    for i in range(config['N_PIXELS']):
        strip.setPixelColorRGB(i, int(pixels[0][i]), int(pixels[0][i]), int(pixels[0][i]))

    strip.show()

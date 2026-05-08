import os
# print(os.environ['PYOPENGL_PLATFORM'])
os.environ['DISPLAY'] = ':1'

import pyglet
pyglet.window.Window(100,100)

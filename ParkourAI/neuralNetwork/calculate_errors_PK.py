##################################################################################################################
# used to monitor data output

from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
##################################################################################################################
import nnfs
import os
import pickle # allows us to save the parameters of the model into a file.
import copy
#from nnfs.datasets import spiral_data, vertical_data, sine_data #imported data set from nnfs.datasets
import cv2

try:
     import numpy as np
except: 
     # Install numpy if they do not have it installed already
     import pip 
     pip.main(['install', 'numpy'])
     import numpy as np
'''
try:
     import cv2    
except:
     # Installs opencv if you do not have it installed already, 
     # NB: opencv download can take 30 - 60 mins.
     import pip
     pip.main(['install', 'opencv-python']) 
     import cv2
except:
     # if this dose not work then try conda
     import subprocess
     subprocess.run(["conda", "install", "-c", "conda-forge", "opencv"])
     import cv2
'''

nnfs.init()





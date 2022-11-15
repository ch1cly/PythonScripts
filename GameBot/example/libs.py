# Базовые Модули
import time    # модуль для операций со временными характеристиками
import random
import numpy as np

# Модули Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Модули Keras-RL2
import rl.core as krl
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# Модули визуализации
from celluloid import Camera
import matplotlib.pyplot as plt
from matplotlib import rc
rc('animation', html='jshtml')
#%matplotlib inline
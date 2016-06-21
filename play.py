from __future__ import division
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
from os.path import isfile
import time
from collections import deque

NUM_ACTIONS = 2
NUM_STATES = 4
WEIGHT_FILE = 'weights.h5'

def create_model(n_inputs, n_outputs):
  model = Sequential([
    Dense(6, batch_input_shape = (None, n_inputs)),
    Activation('relu'),
    Dense(6),
    Activation('relu'),
    Dense(n_outputs),
    Activation('softmax')
  ])
  model.compile('adam', loss = 'mse')
  if isfile(WEIGHT_FILE):
    print "[+] Loaded weights from file"
    model.load_weights(WEIGHT_FILE)
  return model

env = gym.make('CartPole-v0')
model = create_model(NUM_STATES, NUM_ACTIONS)

for index_game in range(100):
  observation = env.reset()
  while True:
    env.render()
    q = model.predict(observation.reshape(1, NUM_STATES))
    action = np.argmax(q)
    observation, reward, done, info = env.step(action)
    time.sleep(0.05)
    if done:
      break
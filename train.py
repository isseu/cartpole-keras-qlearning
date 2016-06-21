from __future__ import division
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import sgd
import os
import random
from os.path import isfile
from collections import deque

NUM_ACTIONS = 2
NUM_STATES = 4
MAX_REPLAY_STATES = 100
BATCH_SIZE = 20
NUM_GAMES_TRAIN = 500
JUMP_FPS = 2
WEIGHT_FILE = 'weights.h5'

def create_model(n_inputs, n_outputs):
  model = Sequential([
    Dense(8, batch_input_shape = (None, n_inputs)),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(n_outputs),
    Activation('linear')
  ])
  model.compile('adam', loss = 'mse')
  if isfile(WEIGHT_FILE):
    print "[+] Loaded weights from file"
    model.load_weights(WEIGHT_FILE)
  return model

env = gym.make('CartPole-v0')
env.monitor.start('training', force = True)
model = create_model(NUM_STATES, NUM_ACTIONS)

replay = []

gamma = 0.99
epsilon = 1
for number_game in range(NUM_GAMES_TRAIN):
  new_state = env.reset()
  reward_game = 0
  done = False
  loss = 0
  index_train_per_game = 0
  print '[+] Starting Game ' + str(number_game)
  while not done:
    env.render()
    index_train_per_game += 1
    if random.random() < epsilon:
      action = np.random.randint(NUM_ACTIONS)
    else:
      q = model.predict(new_state.reshape(1, NUM_STATES))[0]
      action = np.argmax(q)
    old_state = new_state
    new_state, reward, done, info = env.step(action)
    reward_game += reward
    replay.append([new_state, reward, action, done, old_state])
    if len(replay) > MAX_REPLAY_STATES: replay.pop(np.random.randint(MAX_REPLAY_STATES) + 1)
    if JUMP_FPS != 1 and index_train_per_game % JUMP_FPS == 0: # We skip this train, but already add data
      continue
    len_mini_batch = min(len(replay), BATCH_SIZE)
    mini_batch = random.sample(replay, len_mini_batch)
    X_train = np.zeros((len_mini_batch, NUM_STATES))
    Y_train = np.zeros((len_mini_batch, NUM_ACTIONS))
    for index_rep in range(len_mini_batch):
      new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = mini_batch[index_rep]
      old_q = model.predict(old_rep_state.reshape(1, NUM_STATES))[0]
      new_q = model.predict(new_rep_state.reshape(1, NUM_STATES))[0]
      update_target = np.copy(old_q)
      if done_rep:
        update_target[action_rep] = -1
      else:
        update_target[action_rep] = reward_rep + (gamma * np.max(new_q))
      X_train[index_rep] = old_rep_state
      Y_train[index_rep] = update_target
    loss += model.train_on_batch(X_train, Y_train)
    if reward_game > 200:
      break
  print "[+] End Game {} | Reward {} | Epsilon {:.4f} | TrainPerGame {} | Loss {:.4f} ".format(number_game, reward_game, epsilon, index_train_per_game, loss / index_train_per_game * JUMP_FPS)
  if epsilon >= 0.1:
    epsilon -= (1 / (NUM_GAMES_TRAIN))
  if isfile(WEIGHT_FILE):
    os.remove(WEIGHT_FILE)
  model.save_weights(WEIGHT_FILE)
env.monitor.close()
gym.upload( 'training', api_key = '' )

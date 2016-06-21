from __future__ import division
import gym
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn import lstm, embedding
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam
import os
import random
from os.path import isfile
from collections import deque

NUM_ACTIONS = 2
NUM_STATES = 4
MAX_REPLAY_STATES = 1000
BATCH_SIZE = 50
NUM_GAMES_TRAIN = 1500
WEIGHT_FILE = 'weights.h5'

def create_model(n_inputs, n_outputs):
  network = input_data(shape = [None, n_inputs])
  network = fully_connected(network, 16, activation = 'relu')
  network = dropout(network, 0.5)
  network = fully_connected(network, 32, activation = 'relu')
  network = dropout(network, 0.5)
  network = fully_connected(network, n_outputs, activation = 'linear')
  network = regression(network,
  optimizer = 'adam',
  loss = 'softmax_categorical_crossentropy')
  model = tflearn.DNN(
    network,
    max_checkpoints = 0,
    tensorboard_verbose = 0,
    tensorboard_dir = 'logs'
  )
  return model

env = gym.make('CartPole-v0')
# env.monitor.start('training')
model = create_model(NUM_STATES, NUM_ACTIONS)

replay = deque([])

gamma = 0.99
epsilon = 1
for number_game in range(NUM_GAMES_TRAIN):
  new_state = env.reset()
  reward_game = 0
  print '[+] Starting Game ' + str(number_game)
  while True:
    env.render()
    q = model.predict([new_state])[0]
    if random.random() < epsilon:
      action = np.random.randint(0, NUM_ACTIONS)
    else:
      action = np.argmax(q)
    old_state = new_state
    new_state, reward, done, info = env.step(action)
    reward_game += reward
    replay.append((new_state, reward, action, done, old_state))
    if len(replay) > MAX_REPLAY_STATES: replay.popleft()
    mini_batch = random.sample(replay, min(len(replay), BATCH_SIZE))
    X_train = np.zeros((BATCH_SIZE, NUM_STATES))
    Y_train = np.zeros((BATCH_SIZE, NUM_ACTIONS))
    for index_rep in range(len(mini_batch)):
      new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = mini_batch[index_rep]
      print new_rep_state, reward_rep, action_rep, done_rep, old_rep_state 
      old_q = model.predict([old_rep_state])[0]
      new_q = model.predict([new_rep_state])[0]
      print new_q
      max_new_q = np.max(new_q)
      update_target = np.zeros(NUM_ACTIONS)
      update_target[:] = old_q[:]
      if done_rep:
        update = -1
      else:
        update = (reward_rep + (gamma * max_new_q))
        print update, reward_rep, gamma, max_new_q
      update_target[action_rep] = update
      X_train[index_rep] = old_rep_state
      Y_train[index_rep] = update_target
    model.fit(
      X_train, Y_train,
      validation_set = 0,
      n_epoch = 1,
      batch_size = MAX_REPLAY_STATES,
      shuffle = True,
      show_metric = False,
      snapshot_step = 200,
      snapshot_epoch = False,
      run_id = 'carpole_rl'
    )
    if done or reward_game > 200:
      break
  print "[+] End Game " + str(number_game) + ", Reward " + str(reward_game) + ", Epsilon " + str(epsilon)
  if epsilon > 0.05:
    epsilon -= (1 / NUM_GAMES_TRAIN)
  if isfile(WEIGHT_FILE):
    os.remove(WEIGHT_FILE)
  model.save(WEIGHT_FILE)
env.monitor.close()
gym.upload(
 '/tmp/cartpole-experiment-1',
 writeup = 'https://gist.github.com/isseu/7c295d4d2b46e5d9a18dd845ef07dcb9',
 api_key = ''
)
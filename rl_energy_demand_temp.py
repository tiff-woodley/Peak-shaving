# code adapted from stock trading rl agent tutorial found at:
# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python

# package imports
from __future__ import print_function, division
from builtins import range
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv


# global counter
global_iters = 0

## Functions to read in data
def get_data():
  # 2019  energy demand data without outliers
  df = pd.read_csv('2019_unormalised.csv')
  return df.values

def get_data_test():
  # 2019 energy demand data with outliers
  df = pd.read_csv('2019.csv')
  return df.values

def get_data_test_two():
  # 2018 energy demand data
  df = pd.read_csv('2018.csv')
  return df.values

def get_cluster():
    # 2019 energy demand cluster number without outliers
    df = pd.read_csv("2019_clusters.csv")
    return df.values.flatten()

def get_test_cluster():
    # 2019 energy demand cluster number with outliers
    df = pd.read_csv("2019_clusters_full.csv")
    return df.values.flatten()


def get_test_cluster_two():
    # 2018 energy demand cluster number
    df = pd.read_csv("2018_clusters.csv")
    return df.values.flatten()


def get_temp():
    # 2019 temperature data with outliers
    df = pd.read_csv("2019_train_temp.csv")
    return df.values.flatten()

def get_test_temp():
    # 2019 temperate data without outliers
    df = pd.read_csv("temp_2019_max.csv")
    return df.values.flatten()

def get_test_temp_two():
    # 2018 temperature data
    df = pd.read_csv("temp_2018_max.csv")
    return df.values.flatten()

# Create parh directories if they do not exsist
def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


# Object to set up the environment
class EnergyDemandEnv:

  def __init__(self, data, cluster, test_cluster, test_cluster_two, which_cluster, test_data,test_data_two, capacities, temp, temp_test, temp_test_two):

    self.profile_history = data
    self.test_history = test_data
    self.test_history_two = test_data_two
    self.clusters = cluster
    self.in_test = 1
    self.test_no = 0
    self.n_episodes, self.n_timesteps = self.profile_history.shape
    self.episode_no = 1 
    self.episode = self.profile_history[self.episode_no,]
    self.initial_capacity = 800000
    self.cur_step = None
    self.capacity = None
    self.maxvalue = None
    self.outputs = None
    self.cumrewards = None 
    self.previousreward = None
    self.cluster_no = cluster[self.episode_no]
    self.capacity = self.initial_capacity
    self.current_temp = None
    
    # filtering data for chosen cluster
    cluster_agent = which_cluster
    self.profile_history = data[cluster == cluster_agent,]
    self.test_history = test_data[test_cluster == cluster_agent,]
    self.test_history_two = test_data_two[test_cluster_two == cluster_agent,]
    self.temp = temp[cluster == cluster_agent,]
    self.temp_test = temp_test[test_cluster == cluster_agent,]
    self.temp_test_two = temp_test_two[test_cluster_two == cluster_agent,]
    
    # calculating cluster average profile
    average_arrays = []

    for k in range(1,4):
        average_array = []
        for i in range(0,data.shape[1]):
            total = 0
            count = 0
            for j in range(0,data.shape[0]):
                      
                if(cluster[j] == k):    
                    total = total + data[j,i]
                    count = count + 1
                
            average_array.append(total/count)
        average_arrays.append(average_array)  
        
    self.avg_episode = average_arrays[cluster_agent - 1]  
    
    plt.plot(self.avg_episode)
    plt.show()

    # setting up the action space
    self.action_space = np.arange(5)
    self.action_list = np.array([0,50000,100000,150000,200000])

    # number of agent states
    self.state_dim = 4
    
    # reeset the environment
    self.reset()


  def reset(self):
    self.cur_step = 0
    self.maxvalue = 0
    self.outputs = np.zeros(48)
    self.cumrewards = 0

    if self.in_test == 1:
        # if in training, sample profile from training data and reset the battery to full capacity
        self.episode_no = random.sample(range(1,self.profile_history.shape[0]),1)
        self.capacity = self.initial_capacity      
        self.episode = self.profile_history[self.episode_no[0],]
        self.current_temp = self.temp[self.episode_no[0],]

    elif self.in_test == 2:
        # if in testing on 2019, sample profile from 2019 data and reset the battery to full capacity
        self.episode = self.test_history[self.test_no,]
        self.current_temp = self.temp_test[self.test_no,]
        self.capacity = self.initial_capacity 
        self.test_no = self.test_no + 1
    else:
        # if in testing on 2018, sample profile from 2018 data and reset the battery to full capacity
        self.episode = self.test_history_two[self.test_no,]
        self.current_temp = self.temp_test_two[self.test_no,]
        self.capacity = self.initial_capacity 
        self.test_no = self.test_no + 1
         
    # calculate optimal day ahead reduction on average profile    
    self.previousreward = self.curve_max_dispatch()    

    # returns the initial state
    return self._get_obs()


  
  def curve_max_dispatch(self):
      
      # works out the max that could be reduced from current time step going forward - on the average profile
        
     level = np.max(self.avg_episode) 

     curve_sliced = self.avg_episode
     curve_sliced = curve_sliced[self.cur_step:48]
     
     curve_before = self.avg_episode - self.outputs
     
     if self.cur_step > 0:
         curve_before = curve_before[0:self.cur_step]
         max_reached = np.max(curve_before)
     else:
         max_reached = curve_before[0]
        
     while (True):
         output = curve_sliced - np.repeat(level,48 - self.cur_step)
         output[output<0]= 0
         # integrate the area of output
         kwh_used = np.sum(output/2)
         if kwh_used < self.capacity:
             level = level - 1000
         else:
             break
     if np.max(curve_sliced - output) > max_reached:
         return np.max(self.avg_episode) - np.max(curve_sliced - output) 
     return np.max(self.avg_episode) - max_reached     
  
 
      
  def step(self, action):
    
    # make sure action is in the action space  
    assert action in self.action_space

    # perform the action
    self._dispatch(action)

    # get the new demand after taking the action
    cur_demand = self.episode[self.cur_step] - self.outputs[self.cur_step]
    
    # done if we have run out of data
    done = self.cur_step == 47
        
    reward = 0
    
    # reset max demand if been exceeded
    if (cur_demand >= self.maxvalue):
        self.maxvalue = cur_demand   
    
    if (done):
        max_before = np.max(self.episode)
        max_after = self.maxvalue
        # positive reward for amount saved
        reward = reward + (max_before - max_after)/1000

    # negative reward if potential is reduced
    current_reward = self.curve_max_dispatch()
    reward = reward - (self.previousreward - current_reward )/1000

    #update for next step
    self.previousreward =  current_reward   
    self.cur_step += 1
    self.cumrewards = self.cumrewards + reward
    
    # store the current value of the max reached here
    info = {'cur_max': self.maxvalue, 'cur_reward':self.cumrewards}

    # conform to the Gym API
    return self._get_obs(), reward, done, info

  # returns current state
  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[0] = self.cur_step
    obs[1] = self.capacity 
    obs[2] = self.maxvalue
    obs[3] = self.current_temp
    
    scaled_obs = obs
    scaled_obs[0] = scaled_obs[0]/48
    scaled_obs[1] = scaled_obs[1]/self.initial_capacity
    scaled_obs[2] = scaled_obs[2]/np.max(self.episode)
    scaled_obs[3] = (scaled_obs[3] - 14)/(36-14)
    
    return scaled_obs
    
  # Take the given action
  def _dispatch(self, action):
    
    # Get the value of the action  
    action_value = self.action_list[action]
    
    #check if there is enough capacity, if so add to output
    if (action_value/2 < self.capacity):
        self.capacity = self.capacity - action_value/2 
        output = action_value
    else:
        output = self.capacity*2
        self.capacity = 0
        
    self.outputs[self.cur_step] = output    


# a version of HiddenLayer that keeps track of params
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    # parameter initialisation  
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
      self.params.append(self.b)
    self.f = f
  
  # multiply a layer of the network  
  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)

# create deep q learning network object 
class DQN:
  def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_sz=32):
    self.K = K

    # create the network
    self.layers = []
    # set beginning of first layer to state input size
    M1 = D
    # add on inner layers
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x)
    self.layers.append(layer)

    # collect params for copy
    self.params = []
    for layer in self.layers:
      self.params += layer.params
    tf.print(self.params)  

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z
    self.predict_op = Y_hat

    selected_action_values = tf.reduce_sum(
      Y_hat * tf.one_hot(self.actions, K),
      reduction_indices=[1]
    )

    cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
    self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

    # create replay memory
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def set_session(self, session):
    self.session = session

  def copy_from(self, other):
    # collect all the ops
    ops = []
    my_params = self.params
    other_params = other.params
    for p, q in zip(my_params, other_params):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)
    # now run them all
    self.session.run(ops)

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if len(self.experience['s']) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
    states = [self.experience['s'][i] for i in idx]
    actions = [self.experience['a'][i] for i in idx]
    rewards = [self.experience['r'][i] for i in idx]
    next_states = [self.experience['s2'][i] for i in idx]
    dones = [self.experience['done'][i] for i in idx]
    next_Q = np.max(target_network.predict(next_states), axis=1)
    targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

    # call optimizer
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )

  def add_experience(self, s, a, r, s2, done):
    if len(self.experience['s']) >= self.max_experiences:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      X = np.atleast_2d(x)
      return np.argmax(self.predict(X)[0])


# function to run through a training episode
def play_one(env, model, tmodel, eps, gamma, copy_period):
  global global_iters
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    totalreward += reward

    # update the model
    model.add_experience(prev_observation, action, reward, observation, done)
    model.train(tmodel)

    iters += 1
    global_iters += 1

    if global_iters % copy_period == 0:
      tmodel.copy_from(model)
      
  return totalreward

# function to run through a test episode
def play_one_test(env, model, eps):
  global global_iters
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done:

    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    totalreward += reward    

  return totalreward


def main():
  
  # fetching data 
  data = get_data()
  cluster = get_cluster()
  test_cluster = get_test_cluster()
  test_data = get_data_test()
  test_cluster_two = get_test_cluster_two()
  test_data_two = get_data_test_two()
  capacities = get_capacity()    
  temp = get_temp()
  temp_test = get_test_temp()
  temp_test_two = get_test_temp_two()
  n_episodes, n_timesteps = data.shape

  # used to choose which cluster to run on
  which_cluster = 3
  
  # initialising the environment
  env = EnergyDemandEnv(data, cluster, test_cluster,test_cluster_two, which_cluster,test_data,test_data_two, capacities, temp, temp_test, temp_test_two)
  
  # initialising the neural network
  gamma = 0.99
  copy_period = 50 
  state_size = env.state_dim
  action_size = len(env.action_space)
  sizes = [10,10]
  model = DQN(state_size, action_size, sizes, gamma)
  tmodel = DQN(state_size, action_size, sizes, gamma)
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  model.set_session(session)
  tmodel.set_session(session)

  
  # loop to run 5000 training episodes
  N = 5000
  i = 0
  totalrewards = np.empty(N)
  totalrewardsroll = np.empty(250)

  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 20 == 0:
      totalrewardsroll[i] = totalrewards[max(0, n-20):(n+1)].mean() 
      i = i + 1
      print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 20):", totalrewards[max(0, n-20):(n+1)].mean())

      if True:  
          plt.plot(env.episode)
          plt.plot(env.outputs)
          plt.plot(env.episode - env.outputs)
          plt.show() 


          
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())
  
  # loop to test on the 2019 data - epsilon set to zero, no updates to neural network
  if (True):
      
      results = np.zeros(len(test_cluster))
      eps = 0
      N = len(test_cluster[test_cluster == which_cluster])
      maxvalues = np.empty(N)
      env.in_test = 2
      for n in range(N):
          totalreward = play_one_test(env,tmodel,eps)
          print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 20):", totalrewards[max(0, n-20):(n+1)].mean())
          
          maxvalues[n] = round(np.max(env.episode - env.outputs),2)
          
          if True:  
              plt.plot(env.episode)
              plt.plot(env.outputs)
              plt.plot(env.episode - env.outputs)
              plt.text(0,1000000,str(n)+ " 2019")
              plt.show() 
        
 
      count = 0        
      for i in range(0,len(test_cluster)):
          
          if test_cluster[i] == which_cluster:
              results[i] = maxvalues[count]
              count = count + 1
                            
              
      pd.DataFrame(results).to_csv("cluster_results_2019.csv")

  # loop to test on the 2018 data - epsilon set to zero, no updates to neural network    
  if (True):
      
      results = np.zeros(len(test_cluster_two))
      eps = 0
      N = len(test_cluster[test_cluster_two == which_cluster])
      maxvalues = np.empty(N)
      env.in_test = 3
      env.test_no = 0
      for n in range(N):
          totalreward = play_one_test(env,tmodel,eps)
          print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 20):", totalrewards[max(0, n-20):(n+1)].mean())
          
          maxvalues[n] = round(np.max(env.episode - env.outputs),2)
          
          if True:  
              plt.plot(env.episode)
              plt.plot(env.outputs)
              plt.plot(env.episode - env.outputs)
              plt.text(0,1000000,str(n)+ " 2018")
              plt.show() 
        
 
      count = 0        
      for i in range(0,len(test_cluster_two)):
          
          if test_cluster_two[i] == which_cluster:
              results[i] = maxvalues[count]
              count = count + 1
                            
              
      pd.DataFrame(results).to_csv("cluster_results_2018.csv")      

  
  session.close()
  
  plt.plot(totalrewardsroll)
  plt.title("Rewards")
  plt.show()


if __name__ == '__main__':
  main()
  
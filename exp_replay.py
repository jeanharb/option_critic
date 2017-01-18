"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
  """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

  """
  def __init__(self, width, height, rng, max_steps=1000, phi_length=4):
    """Construct a DataSet.

    Arguments:
      width, height - image size
      max_steps - the number of time steps to store
      phi_length - number of images to concatenate into a state
      rng - initialized numpy random number generator, used to
      choose random minibatches

    """
    # TODO: Specify capacity in number of state transitions, not
    # number of saved time steps.

    # Store arguments.
    self.width = width
    self.height = height
    self.max_steps = max_steps
    self.phi_length = phi_length
    self.rng = rng

    # Allocate the circular buffers and indices.
    self.imgs = np.zeros((max_steps, height, width), dtype='uint8')
    self.actions = np.zeros(max_steps, dtype='int32')
    self.rewards = np.zeros(max_steps, dtype=floatX)
    self.terminal = np.zeros(max_steps, dtype='bool')
    
    self.bottom = 0
    self.top = 0
    self.size = 0

  def add_sample(self, img, action, reward, terminal):
    """Add a time step record.

    Arguments:
      img -- observed image
      action -- action chosen by the agent
      reward -- reward received after taking the action
      terminal -- boolean indicating whether the episode ended
      after this time step
    """
    self.imgs[self.top] = img
    self.actions[self.top] = action
    self.rewards[self.top] = reward
    self.terminal[self.top] = terminal

    if self.size == self.max_steps:
      self.bottom = (self.bottom + 1) % self.max_steps
    else:
      self.size += 1
    self.top = (self.top + 1) % self.max_steps

  def __len__(self):
    """Return an approximate count of stored state transitions."""
    # TODO: Properly account for indices which can't be used, as in
    # random_batch's check.
    return max(0, self.size - self.phi_length)

  def last_phi(self):
    """Return the most recent phi (sequence of image frames)."""
    indexes = np.arange(self.top - self.phi_length, self.top)
    return self.imgs.take(indexes, axis=0, mode='wrap')

  def phi(self, img):
    """Return a phi (sequence of image frames), using the last phi_length -
    1, plus img.

    """
    indexes = np.arange(self.top - self.phi_length + 1, self.top)

    phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
    phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                          axis=0,
                          mode='wrap')
    phi[-1] = img
    return phi

  def random_batch(self, batch_size, random_selection=False):
    """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

    """
    # Allocate the response.
    states = np.zeros((batch_size,
               self.phi_length,
               self.height,
               self.width),
              dtype='uint8')
    actions = np.zeros((batch_size), dtype='int32')
    rewards = np.zeros((batch_size), dtype=floatX)
    terminal = np.zeros((batch_size), dtype='bool')
    next_states = np.zeros((batch_size,
                self.phi_length,
                self.height,
                self.width),
                 dtype='uint8')

    count = 0
    indices = np.zeros((batch_size), dtype='int32')
      
    while count < batch_size:
      # Randomly choose a time step from the replay memory.
      index = self.rng.randint(self.bottom,
                     self.bottom + self.size - self.phi_length)

      initial_indices = np.arange(index, index + self.phi_length)
      transition_indices = initial_indices + 1
      end_index = index + self.phi_length - 1

      if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
        continue

      indices[count] = index

      # Add the state transition to the response.
      states[count] = self.imgs.take(initial_indices, axis=0, mode='wrap')
      actions[count] = self.actions.take(end_index, mode='wrap')
      rewards[count] = self.rewards.take(end_index, mode='wrap')
      terminal[count] = self.terminal.take(end_index, mode='wrap')
      next_states[count] = self.imgs.take(transition_indices,
                        axis=0,
                        mode='wrap')
      count += 1

    return states, actions, rewards, next_states, terminal

if __name__ == "__main__":
  pass
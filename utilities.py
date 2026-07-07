import numpy as np
import math as math
from scipy.stats import norm, bootstrap
from sklearn.model_selection import KFold
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

BIN_WIDTH = 0.01 # in seconds
MAX_TIME_STEPS = 100 # in bins
Array = Any
PRNGKey = Any

def mask_sequences(sequence_batch: Array, lengths: Array) -> Array:
  """Sets positions beyond the length of each sequence to 0."""
  return sequence_batch * (
    lengths[:, None] > np.arange(sequence_batch.shape[1])[None])

def causalgaussian(sigma):
  """Returns a causal gaussian filter with standard deviation sigma."""
  dt = BIN_WIDTH
  maxtimesteps = MAX_TIME_STEPS
  h = norm.pdf(np.linspace(0.0, dt*maxtimesteps-dt, num=maxtimesteps), loc=0, scale=sigma)
  w = 1. / (np.convolve(np.ones(maxtimesteps)*dt, h)[:maxtimesteps])
  return h, w

def smooth(y):
  """Smooths a vector y with a causal gaussian filter."""
  h, w = causalgaussian(0.1)
  x = np.convolve(y, h, mode='full')
  return np.array([w[t]*x[t] for t in range(len(y))])

def generate_smoothed_spikes(spikes, lengths):
  """Smooths the spikes with a causal gaussian filter."""
  smoothed_spikes = np.zeros_like(spikes).astype(np.float32)
  for trial in range(smoothed_spikes.shape[0]):
    for neuron in range(smoothed_spikes.shape[2]):
      smoothed_spikes[trial, :lengths[trial], neuron] = smooth(spikes[trial, :lengths[trial], neuron])
  return smoothed_spikes

def generate_psths(spikes, lengths):
  """Generates the PSTHs conditioned on left and right choices of the animal."""
  smoothed_spikes = generate_smoothed_spikes(spikes, lengths)
  mask_ = mask_sequences(np.ones_like(smoothed_spikes[:,:,0]), lengths)
  smoothed_spikes[mask_ == 0] = np.nan
    
  observed_psth = np.nanmean(smoothed_spikes, axis=0)  
  psth_ci_low = np.zeros(observed_psth.shape)    
  psth_ci_high = np.zeros(observed_psth.shape)
    
  num_neurons = smoothed_spikes.shape[2]
  num_timebins = smoothed_spikes.shape[1]
    
  for neuron in range(num_neurons):
    for timebin in range(num_timebins):
      smoothed = smoothed_spikes[:, timebin, neuron]
      data_ = (smoothed[~np.isnan(smoothed)],)
      res = bootstrap(data_, np.mean, n_resamples=1000, confidence_level=0.95)
      psth_ci_low[timebin, neuron] = res.confidence_interval.low
      psth_ci_high[timebin, neuron] = res.confidence_interval.high

  return observed_psth, psth_ci_low, psth_ci_high

def split_data(data, n_splits, seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_valid_indices = []
    test_indices = []
    np.random.seed(seed=seed)
    
    for i, (train_valid_index, test_index) in enumerate(kf.split(data)):
        train_valid_indices.append(np.random.permutation(train_valid_index))
        test_indices.append(np.random.permutation(test_index))

    train_indices = []
    valid_indices = []
    for i in range(n_splits-1):
        train_indices.append(
            train_valid_indices[i][~np.isin(train_valid_indices[i], test_indices[i+1])]
        )
        valid_indices.append(
            train_valid_indices[i][np.isin(train_valid_indices[i], test_indices[i+1])]
        )
    train_indices.append(
        train_valid_indices[-1][~np.isin(train_valid_indices[-1], test_indices[0])]
    )
    valid_indices.append(
        train_valid_indices[-1][np.isin(train_valid_indices[-1], test_indices[0])]
    )

    return train_indices, valid_indices, test_indices

def loss_function(x, recon_x):
    MSE = (1000*(x - recon_x) ** 2).mean() # Mean Squared Error
    return MSE

def get_conditional_mean(m):
  """Compute the conditional mean E[n | x = m] for a Gaussian mixture model."""
  a_ms = [[-0.87, 0.87], [-0.87, -0.87], [0.87, -0.87], [0.87, 0.87]]
  a_ns = [[-2.31, 2.31], [-2.31, -2.31], [2.31, -2.31], [2.31, 2.31]]
  cov_true = [[0.25, 0, 0, 0], 
      [0, 0.25, 0, 0],
      [0, 0, 0.25, 0],
      [0, 0, 0, 0.25]]

  m = np.asarray(m)
  a_ms = np.asarray(a_ms)
  a_ns = np.asarray(a_ns)

  cov_xx = cov_true[:2, :2]
  cov_yx = cov_true[2:, :2]
  inv_cov_xx = np.linalg.inv(cov_xx)  # compute once

  # Precompute differences (shape: [n, 2])
  diff = a_ms - m

  # Compute Mahalanobis distances efficiently
  md2 = np.einsum('ij,jk,ik->i', diff, inv_cov_xx, diff)
  weights = np.exp(-0.5 * md2)
  weights /= np.sum(weights)  # normalize

  # Compute conditional means for each component
  cond_means = a_ns + (cov_yx @ inv_cov_xx @ diff.T).T  # shape: [n, 1 or d]

  # Weighted sum (conditional mean)
  mean_val = np.sum(weights[:, None] * cond_means, axis=0)
  return mean_val.squeeze()
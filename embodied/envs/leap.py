"""Offline Leap+XELA env stub.

Used only to define DreamerV3 observation/action spaces for world-model
training on converted finger-close (or other grasp) episodes. Steps return
zeros; the policy is not evaluated in the real simulator.
"""

import functools

import embodied
import numpy as np


N_JOINTS = 16
N_TAXELS = 368
# Wide enough to cover Leap actuator ctrlrange in the MuJoCo scene / URDF.
ACTION_LOW = -1.5
ACTION_HIGH = 2.5


def normalize_action(action, low=ACTION_LOW, high=ACTION_HIGH):
  action = np.asarray(action, dtype=np.float32)
  return np.clip(2.0 * (action - low) / (high - low) - 1.0, -1.0, 1.0)


class Leap(embodied.Env):

  def __init__(self, task='finger_close', length=64, **kwargs):
    del kwargs
    self._task = task
    self._length = int(length)
    self._step = 0
    self._done = True

  @functools.cached_property
  def obs_space(self):
    taxel = (N_TAXELS, 3)
    return {
        'forces_local': embodied.Space(np.float32, taxel),
        'positions': embodied.Space(np.float32, taxel),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    return {
        'action': embodied.Space(
            np.float32, (N_JOINTS,), ACTION_LOW, ACTION_HIGH),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      return self._obs(is_first=True)
    self._step += 1
    self._done = self._step >= self._length
    return self._obs(is_last=self._done, is_terminal=self._done)

  def _obs(self, is_first=False, is_last=False, is_terminal=False):
    zeros = np.zeros((N_TAXELS, 3), np.float32)
    return {
        'forces_local': zeros,
        'positions': zeros.copy(),
        'reward': np.float32(0.0),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

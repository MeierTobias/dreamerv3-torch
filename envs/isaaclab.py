import datetime
import uuid

import gym
import numpy as np
import torch


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab DirectRLEnv for use with DreamerV3.

    Keeps data as torch GPU tensors.  Manages per-env episode UUIDs and
    tracks auto-resets to produce ``is_first`` / ``is_terminal`` signals.

    The wrapper is a pure API adapter — it performs **no** data
    transformation.  All observation keys from the env are passed through
    unchanged.  The env should already produce observations in the format
    that DreamerV3 expects (e.g. uint8 images at the correct resolution,
    float32 state vectors).  Use the DreamerV3 config's ``cnn_keys`` /
    ``mlp_keys`` regex to tell the encoder which keys to route where.

    All task-specific customisation (custom rewards, extra obs keys, camera
    images, etc.) should be applied by monkey-patching the env **before**
    wrapping it with this class.
    """

    metadata = {}

    def __init__(self, env):
        """
        Args:
            env: IsaacLab env instance, possibly wrapped with gymnasium Wrappers.
        """
        self._env = env
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self._num_envs = unwrapped.num_envs
        self._device = unwrapped.device
        self._done = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._ids = [self._make_id() for _ in range(self._num_envs)]

    @property
    def observation_space(self):
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        spaces = {}
        for key, box in unwrapped.single_observation_space.spaces.items():
            spaces[key] = gym.spaces.Box(
                low=float(box.low.flat[0]),
                high=float(box.high.flat[0]),
                shape=box.shape,
                dtype=box.dtype,
            )
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        space = unwrapped.single_action_space
        low = np.clip(space.low, -1.0, 1.0).astype(np.float32)
        high = np.clip(space.high, -1.0, 1.0).astype(np.float32)
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def ids(self):
        return self._ids

    def reset(self):
        obs_dict, _ = self._env.reset()
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._done = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._ids = [self._make_id() for _ in range(self._num_envs)]
        return self._make_obs(obs_dict)

    def step(self, action):
        obs_dict, reward, terminated, truncated, extras = self._env.step(action)
        done = terminated | truncated

        obs = self._make_obs(obs_dict, terminated=terminated)
        info = {
            "discount": (~terminated).float(),
            "terminated": terminated,
            "truncated": truncated,
        }

        # prepare is_first for the *next* step: envs that are done now
        # will have auto-reset, so their next obs is a first obs
        self._is_first = done.clone()
        # assign new UUIDs for auto-reset envs
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            self._ids[i] = self._make_id()

        self._done = done
        return obs, reward, done, info

    def _make_obs(self, obs_dict, terminated=None):
        obs = {}
        for key, val in obs_dict.items():
            obs[key] = val
        obs["is_first"] = self._is_first
        if terminated is not None:
            obs["is_terminal"] = terminated
        else:
            obs["is_terminal"] = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        return obs

    def close(self):
        self._env.close()

    @staticmethod
    def _make_id():
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

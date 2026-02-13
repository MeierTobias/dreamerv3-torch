import datetime
import uuid

import gym
import numpy as np
import torch


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab DirectRLEnv for use with DreamerV3.

    Keeps data as torch GPU tensors. Manages per-env episode UUIDs and
    tracks auto-resets to produce is_first/is_terminal signals.

    The wrapper does not decide what is vision vs proprio — it exposes
    all observation keys and lets the encoder config (mlp_keys/cnn_keys)
    control what gets used, same as the DMC env.
    """

    metadata = {}

    def __init__(self, env, obs_key="policy", obs_names=None, image_key=None, size=(64, 64)):
        """
        Args:
            env: IsaacLab DirectRLEnv instance.
            obs_key: key in the env's obs dict to read from (default "policy").
            obs_names: dict mapping {name: dim} to split a vector obs into
                named keys for the MLP encoder. If None and image_key is None,
                the raw tensor is exposed as "obs".
            image_key: if set, expose the obs under this key name (e.g. "image")
                and convert to uint8. Used for vision envs.
            size: target (H, W) for image resize when image_key is set.
        """
        self._env = env
        self._obs_key = obs_key
        self._obs_names = obs_names
        self._image_key = image_key
        self._size = size
        self._num_envs = env.num_envs
        self._device = env.device
        self._done = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_first = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._ids = [self._make_id() for _ in range(self._num_envs)]

    @property
    def single_obs_shape(self):
        return self._env.single_observation_space[self._obs_key].shape

    @property
    def observation_space(self):
        obs_shape = self.single_obs_shape
        spaces = {}
        if self._image_key:
            c = obs_shape[-1] if len(obs_shape) == 3 else 3
            spaces[self._image_key] = gym.spaces.Box(
                0, 255, self._size + (c,), dtype=np.uint8
            )
        elif self._obs_names:
            for name, dim in self._obs_names.items():
                spaces[name] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        else:
            spaces["obs"] = gym.spaces.Box(
                -np.inf, np.inf, obs_shape, dtype=np.float32
            )
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.single_action_space
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
        raw = obs_dict[self._obs_key]
        obs = {}
        if self._image_key:
            obs[self._image_key] = self._process_image(raw)
        elif self._obs_names:
            idx = 0
            for name, dim in self._obs_names.items():
                obs[name] = raw[:, idx : idx + dim]
                idx += dim
        else:
            obs["obs"] = raw
        obs["is_first"] = self._is_first
        if terminated is not None:
            obs["is_terminal"] = terminated
        else:
            obs["is_terminal"] = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self._device
            )
        return obs

    def _process_image(self, raw):
        # IsaacLab camera: (N, H, W, C) float normalized or (N, H, W, C) uint8
        if raw.dtype == torch.float32:
            img = (raw.clamp(0, 1) * 255).to(torch.uint8)
        else:
            img = raw
        # resize if needed
        h, w = self._size
        if img.shape[1] != h or img.shape[2] != w:
            # (N,H,W,C) -> (N,C,H,W) for interpolate, then back
            img = img.permute(0, 3, 1, 2).float()
            img = torch.nn.functional.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
            img = img.permute(0, 2, 3, 1).to(torch.uint8)
        return img

    def close(self):
        self._env.close()

    @staticmethod
    def _make_id():
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

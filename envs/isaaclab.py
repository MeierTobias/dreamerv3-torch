import datetime
import math
import uuid

import gym
import numpy as np
import torch


def _torch_gaussian_tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    """Torch equivalent of dm_control's rewards.tolerance with gaussian sigmoid.

    Returns 1 when x falls inside bounds, decays smoothly outside.
    """
    lower, upper = bounds
    in_bounds = (x >= lower) & (x <= upper)
    if margin == 0:
        return torch.where(in_bounds, 1.0, 0.0)
    d = torch.where(x < lower, lower - x, x - upper) / margin
    # gaussian sigmoid: value_at_margin^(d^2)
    scale = math.sqrt(-2 * math.log(value_at_margin))
    value = torch.exp(-0.5 * (d * scale) ** 2)
    return torch.where(in_bounds, 1.0, value)


def _torch_quadratic_tolerance(x, margin=1.0, value_at_margin=0.0):
    """Torch equivalent of dm_control's tolerance with quadratic sigmoid.

    Returns 1 at x==0, decays quadratically, reaches 0 at margin (when
    value_at_margin=0).
    """
    # scale = sqrt(1 - value_at_margin)
    scale = math.sqrt(1.0 - value_at_margin)
    d = torch.abs(x) / margin
    scaled = d * scale
    return torch.where(scaled.abs() < 1.0, 1.0 - scaled**2, torch.zeros_like(x))


def compute_dmc_balance_reward(pole_angle, pole_ang_vel, cart_pos, action):
    """Compute the exact DMC cartpole balance (smooth) reward in torch.

    This replicates dm_control/suite/cartpole.py  Balance._get_reward(sparse=False):
        upright = (cos(pole_angle) + 1) / 2
        centered = (1 + tolerance(cart_pos, margin=2)) / 2
        small_control = (4 + tolerance(action, margin=1, v@m=0, quadratic)) / 5
        small_velocity = (1 + tolerance(ang_vel, margin=5)) / 2
        reward = upright * centered * small_control * small_velocity

    All inputs are (num_envs,) tensors.
    Returns (num_envs,) tensor with reward in [0, 1].
    """
    # upright: (cos(theta) + 1) / 2  --  1 when pole vertical, 0 when inverted
    upright = (torch.cos(pole_angle) + 1.0) / 2.0

    # centered: gaussian tolerance on cart position, bounds=(0,0), margin=2
    centered = _torch_gaussian_tolerance(cart_pos, bounds=(0.0, 0.0), margin=2.0)
    centered = (1.0 + centered) / 2.0

    # small_control: quadratic tolerance on action, margin=1, value_at_margin=0
    small_control = _torch_quadratic_tolerance(action, margin=1.0, value_at_margin=0.0)
    small_control = (4.0 + small_control) / 5.0

    # small_velocity: gaussian tolerance on angular vel, bounds=(0,0), margin=5
    small_velocity = _torch_gaussian_tolerance(
        pole_ang_vel, bounds=(0.0, 0.0), margin=5.0
    )
    small_velocity = (1.0 + small_velocity) / 2.0

    return upright * centered * small_control * small_velocity


class IsaacLabVecEnv:
    """Wraps a vectorized IsaacLab DirectRLEnv for use with DreamerV3.

    Keeps data as torch GPU tensors. Manages per-env episode UUIDs and
    tracks auto-resets to produce is_first/is_terminal signals.

    The wrapper does not decide what is vision vs proprio — it exposes
    all observation keys and lets the encoder config (mlp_keys/cnn_keys)
    control what gets used, same as the DMC env.
    """

    metadata = {}

    def __init__(
        self,
        env,
        obs_key="policy",
        obs_names=None,
        image_key=None,
        size=(64, 64),
        reward_fn=None,
        action_repeat=1,
        disable_termination=False,
        obs_transform=None,
    ):
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
            reward_fn: optional string selecting a custom reward function.
                "dmc_balance" — exact DMC cartpole balance (smooth) reward.
                None — use the environment's native reward (default).
            action_repeat: number of physics sub-steps per agent step. Used to
                scale the custom reward to match DMC's convention of
                accumulating reward over action_repeat sub-steps.
            disable_termination: if True, monkey-patch the underlying env's
                _get_dones() to never signal early termination. Episodes will
                only end via time-based truncation, matching DMC behaviour.
            obs_transform: optional string selecting an observation transform.
                "dmc_cartpole" — remap IsaacLab's [pole_angle, pole_vel,
                cart_pos, cart_vel] to DMC's [cart_pos, cos(θ), sin(θ),
                cart_vel, pole_vel] with matching key names.
                None — use obs_names as-is (default).
        """
        self._env = env
        self._obs_key = obs_key
        self._obs_names = obs_names
        self._image_key = image_key
        self._size = size
        self._reward_fn = reward_fn
        self._action_repeat = action_repeat
        self._obs_transform = obs_transform
        self._num_envs = env.num_envs
        self._device = env.device
        self._done = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_first = torch.ones(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._ids = [self._make_id() for _ in range(self._num_envs)]

        if disable_termination:
            self._patch_no_termination(env)

    @staticmethod
    def _patch_no_termination(env):
        """Monkey-patch _get_dones so terminated is always False.

        This makes the env behave like DMC: episodes only end via time-based
        truncation (max_episode_length), never via early failure. The original
        _get_dones is kept to still compute time_out correctly.
        """
        original_get_dones = env._get_dones

        def _no_termination_get_dones(self):
            terminated, time_out = original_get_dones()
            # suppress all early terminations; keep only time-based truncation
            terminated = torch.zeros_like(terminated)
            return terminated, time_out

        import types

        env._get_dones = types.MethodType(_no_termination_get_dones, env)

    @property
    def single_obs_shape(self):
        return self._env.single_observation_space[self._obs_key].shape

    @property
    def observation_space(self):
        obs_shape = self.single_obs_shape
        spaces = {}

        # --- state observations ---
        if self._obs_transform == "dmc_cartpole":
            # DMC cartpole obs: position(3) = [cart_x, cos(θ), sin(θ)],
            #                   velocity(2) = [cart_vel, pole_vel]
            spaces["position"] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            spaces["velocity"] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        elif self._obs_names:
            for name, dim in self._obs_names.items():
                spaces[name] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        else:
            # fallback: expose the raw vector unless we only have an image
            if not self._image_key:
                spaces["obs"] = gym.spaces.Box(
                    -np.inf, np.inf, obs_shape, dtype=np.float32
                )

        # --- image observation (additive, not exclusive) ---
        if self._image_key:
            c = obs_shape[-1] if len(obs_shape) == 3 else 3
            spaces[self._image_key] = gym.spaces.Box(
                0, 255, self._size + (c,), dtype=np.uint8
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
        self._is_first = torch.ones(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._done = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._ids = [self._make_id() for _ in range(self._num_envs)]
        return self._make_obs(obs_dict)

    def step(self, action):
        obs_dict, reward, terminated, truncated, extras = self._env.step(action)
        done = terminated | truncated

        # override reward with DMC-compatible function if configured
        if self._reward_fn == "dmc_balance":
            uw = self._env  # unwrapped DirectRLEnv
            pole_angle = uw.joint_pos[:, uw._pole_dof_idx[0]]
            pole_ang_vel = uw.joint_vel[:, uw._pole_dof_idx[0]]
            cart_pos = uw.joint_pos[:, uw._cart_dof_idx[0]]
            # action as sent to physics is scaled; get raw [-1,1] action
            raw_action = action[:, 0] if action.dim() > 1 else action
            reward = compute_dmc_balance_reward(
                pole_angle, pole_ang_vel, cart_pos, raw_action
            )
            # DMC accumulates the reward over action_repeat physics sub-steps.
            # IsaacLab computes it once after decimation, so we multiply to
            # match DMC's scale (approximate: assumes reward is ~constant
            # across sub-steps, which is true when the pole is balanced).
            reward = reward * self._action_repeat

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

        # --- state observations ---
        if self._obs_transform == "dmc_cartpole":
            # Read state from the joint data directly. In the RGB camera env
            # obs_dict["policy"] is the image, NOT the state vector, so we
            # must never slice `raw` for scalar joint quantities.
            uw = self._env  # unwrapped DirectRLEnv
            pole_angle = uw.joint_pos[:, uw._pole_dof_idx[0]].unsqueeze(-1)
            pole_vel = uw.joint_vel[:, uw._pole_dof_idx[0]].unsqueeze(-1)
            cart_pos = uw.joint_pos[:, uw._cart_dof_idx[0]].unsqueeze(-1)
            cart_vel = uw.joint_vel[:, uw._cart_dof_idx[0]].unsqueeze(-1)
            # DMC order: position=[cart_x, cos(θ), sin(θ)], velocity=[cart_vel, pole_vel]
            obs["position"] = torch.cat(
                [cart_pos, torch.cos(pole_angle), torch.sin(pole_angle)], dim=-1
            )
            obs["velocity"] = torch.cat([cart_vel, pole_vel], dim=-1)
        elif self._obs_names:
            idx = 0
            for name, dim in self._obs_names.items():
                obs[name] = raw[:, idx : idx + dim]
                idx += dim
        else:
            # fallback: expose the raw vector unless we only have an image
            if not self._image_key:
                obs["obs"] = raw

        # --- image observation (additive, not exclusive) ---
        if self._image_key:
            # For the DMC cartpole comparison we read raw uint8 directly from
            # the tiled camera buffer, because the env's _get_observations()
            # normalises the image (/255 and subtracts spatial mean) which
            # corrupts it for DreamerV3 (whose preprocess() expects raw uint8).
            # For all other envs, use the obs dict and let _process_image
            # handle the dtype conversion.
            if self._reward_fn == "dmc_balance":
                raw_rgb = self._env._tiled_camera.data.output["rgb"]
                obs[self._image_key] = self._process_image(raw_rgb)
            else:
                obs[self._image_key] = self._process_image(raw)

        obs["is_first"] = self._is_first
        if terminated is not None:
            obs["is_terminal"] = terminated
        else:
            obs["is_terminal"] = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self._device
            )
        return obs

    def _process_image(self, raw):
        # Convert to uint8 if the source provides float32 (e.g. [0,1] range)
        if raw.dtype == torch.float32:
            img = (raw.clamp(0.0, 1.0) * 255).to(torch.uint8)
        else:
            img = raw
        # resize if needed to match config.size (e.g. 64x64)
        h, w = self._size
        if img.shape[1] != h or img.shape[2] != w:
            # (N,H,W,C) -> (N,C,H,W) for interpolate, then back
            img = img.permute(0, 3, 1, 2).float()
            img = torch.nn.functional.interpolate(
                img, size=(h, w), mode="bilinear", align_corners=False
            )
            img = img.permute(0, 2, 3, 1).to(torch.uint8)
        return img

    @staticmethod
    def apply_dmc_cartpole_colors(env):
        """Override cart/pole/slider/light colors to match DMC cartpole visuals.

        Must be called after the scene has been created and the simulation
        has been started (i.e. after ``gym.make``). Only touches env_0's
        prims — replicate_physics mirrors them to all other envs.

        DMC cartpole colours (linear RGB):
          - cart & pole ("self" material):     (0.89, 0.65, 0.41)  warm brown
          - slider / rail ("decoration"):      (0.24, 0.47, 0.61)  steel blue
          - dome light → approximate sky:      (0.18, 0.28, 0.37)  blue sky
          - ground plane:                      (0.04, 0.20, 0.31) dark blue-grey
        """
        import isaaclab.sim as sim_utils

        # ---- create materials under /World/Looks ----
        self_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.89, 0.65, 0.41), roughness=0.6, metallic=0.0)
        deco_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.47, 0.61), roughness=0.6, metallic=0.0)
        self_mat_path = "/World/Looks/DmcSelf"
        deco_mat_path = "/World/Looks/DmcDecoration"

        sim_utils.spawn_preview_surface(self_mat_path, self_mat_cfg)
        sim_utils.spawn_preview_surface(deco_mat_path, deco_mat_cfg)

        # ---- bind materials to cart, pole, slider ----
        # Bind to the Xform parent (stronger_than_descendants) so all child
        # visuals/meshes inherit the material.
        env0 = "/World/envs/env_0/Robot"
        for part in ("cart", "pole"):
            sim_utils.bind_visual_material(f"{env0}/{part}", self_mat_path, stronger_than_descendants=True)
        sim_utils.bind_visual_material(f"{env0}/slider", deco_mat_path, stronger_than_descendants=True)

        # ---- dome light → DMC-like sky colour ----
        stage = sim_utils.get_current_stage()
        light_prim = stage.GetPrimAtPath("/World/Light")
        if light_prim.IsValid():
            from pxr import Gf

            light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(0.18, 0.28, 0.37))
            # DMC uses headlight diffuse=0.8 + ambient=0.4, much dimmer than
            # IsaacLab's default 2000. Lower intensity to approximate DMC.
            light_prim.GetAttribute("inputs:intensity").Set(500.0)
            # Make the dome visible as background colour.
            vis_attr = light_prim.GetAttribute("visibleInPrimaryRay")
            if vis_attr:
                vis_attr.Set(True)

        # ---- ground plane ----
        # The cartpole scene has no explicit ground plane prim. We create a
        # large flat box beneath the cart (cart sits at z=2.0) to serve as
        # the floor with the dark blue-grey DMC ground colour.
        ground_cfg = sim_utils.CuboidCfg(
            size=(40.0, 40.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.04, 0.20, 0.31),
                roughness=0.8,
                metallic=0.0,
            ),
        )
        ground_cfg.func(
            "/World/DmcGround",
            ground_cfg,
            translation=(0.0, 0.0, -0.005),
        )

    def close(self):
        self._env.close()

    @staticmethod
    def _make_id():
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

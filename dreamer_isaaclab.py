"""Train DreamerV3 with IsaacLab environments.

Launch IsaacSim before any IsaacLab imports, then run the dreamer training
loop using a vectorized IsaacLab environment.

Usage:
  python dreamer_isaaclab.py --configs isaac_proprio --task isaac_cartpole_balance --logdir /workspace/dreamerv3-torch/logdir/isaac_proprio/isaac_cartpole_balance

  And to see the progress with TensorBoard:
  tensorboard --logdir /workspace/dreamerv3-torch/logdir/isaac_proprio/isaac_cartpole_balance/

  or
  python dreamer_isaaclab.py --configs isaac_vision --task isaac_cartpole_balance_rgb --enable_cameras --logdir /workspace/dreamerv3-torch/logdir/isaac_vision/isaac_cartpole_balance_rgb
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import pathlib

from isaaclab.app import AppLauncher

# parse dreamer + isaaclab args in two stages
parser = argparse.ArgumentParser(description="Train DreamerV3 with IsaacLab.")
parser.add_argument("--configs", nargs="+")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import signal

import gymnasium as gym
import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

# Isaac Sim may override the default SIGINT handler, preventing
# Python's KeyboardInterrupt from firing on Ctrl+C. Restore it
# so that try/finally cleanup (wandb.finish, etc.) works properly.
signal.signal(signal.SIGINT, signal.default_int_handler)

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from envs.isaaclab import IsaacLabVecEnv
from envs.isaac_cartpole_overrides import (
    patch_dmc_cartpole_reward,
    patch_dmc_cartpole_obs,
    patch_no_termination,
    apply_dmc_cartpole_colors,
)
from dreamer import Dreamer, count_steps, make_dataset

# task name -> task configuration dict
# Each entry can specify:
#   gym_id           - registered gymnasium env id
#   pre_wrap_fns     - list of callable(unwrapped_env, config) to patch env before wrapping
#   post_create_fn   - callable(unwrapped_env) to run after gym.make (e.g. colours)
#   env_cfg_fn       - callable(env_cfg, task, config) to customise env_cfg


def _cartpole_env_cfg_fn(env_cfg, task, config):
    """Customise env_cfg for the DMC cartpole benchmark camera variant."""
    if task == "cartpole_balance_rgb" and hasattr(env_cfg, "tiled_camera"):
        # DMC's fixed camera: pos="0 -4 1" (4m back, cart height).
        # IsaacLab cart is at z=2.0, camera looks along +X, so the
        # equivalent is (-4, 0, 2).  Default was (-5, 0, 2).
        env_cfg.tiled_camera.offset.pos = (-4.0, 0.0, 2.0)


def _patch_cartpole_reward(env, config):
    """Pre-wrap callback: patch reward with action_repeat from config."""
    patch_dmc_cartpole_reward(env, config.action_repeat)


def _patch_cartpole_obs(env, config):
    patch_dmc_cartpole_obs(env)


def _patch_no_term(env, config):
    patch_no_termination(env)


ISAAC_TASKS = {
    "cartpole_balance": {
        "gym_id": "Isaac-Cartpole-Direct-v0",
        "pre_wrap_fns": [_patch_no_term, _patch_cartpole_reward, _patch_cartpole_obs],
        "post_create_fn": None,
        "env_cfg_fn": None,
    },
    "cartpole_balance_rgb": {
        "gym_id": "Isaac-Cartpole-RGB-Camera-Direct-v0",
        "pre_wrap_fns": [_patch_no_term, _patch_cartpole_reward, _patch_cartpole_obs],
        "post_create_fn": apply_dmc_cartpole_colors,
        "env_cfg_fn": _cartpole_env_cfg_fn,
    },
}

# import isaaclab tasks to trigger gym.register calls
import isaaclab_tasks  # noqa: F401


def make_isaac_env(config):
    """Create a vectorized IsaacLab environment wrapped for DreamerV3."""
    suite, task = config.task.split("_", 1)
    assert suite == "isaac", f"Expected isaac suite, got {suite}"

    task_info = ISAAC_TASKS.get(task)
    if not task_info:
        avail = ", ".join(f"isaac_{k}" for k in ISAAC_TASKS)
        raise ValueError(f"Unknown isaac task 'isaac_{task}'. Available: {avail}")

    gym_id = task_info["gym_id"]
    pre_wrap_fns = task_info.get("pre_wrap_fns", [])
    post_create_fn = task_info.get("post_create_fn")
    env_cfg_fn = task_info.get("env_cfg_fn")

    num_envs = config.envs
    env_cfg_class = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    # resolve string to class
    if isinstance(env_cfg_class, str):
        module_name, class_name = env_cfg_class.rsplit(":", 1)
        import importlib

        mod = importlib.import_module(module_name)
        env_cfg_class = getattr(mod, class_name)

    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = num_envs
    env_cfg.decimation = config.action_repeat
    env_cfg.seed = config.seed
    # config.time_limit is already in agent steps (divided by action_repeat)
    env_cfg.episode_length_s = config.time_limit * env_cfg.sim.dt * config.action_repeat

    if hasattr(env_cfg, "tiled_camera"):
        render_mode = "rgb_array"
        env_cfg.tiled_camera.width = config.size[1]
        env_cfg.tiled_camera.height = config.size[0]
        from isaaclab.sim import RenderCfg

        env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")
    else:
        render_mode = None

    # apply task-specific env_cfg customisations (e.g. camera position)
    if env_cfg_fn is not None:
        env_cfg_fn(env_cfg, task, config)

    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)

    # apply pre-wrap patches (e.g. disable termination, remap observations)
    for fn in pre_wrap_fns:
        fn(isaac_env.unwrapped, config)

    # apply post-create customisations (e.g. scene colours)
    if post_create_fn is not None:
        post_create_fn(isaac_env.unwrapped)

    vec_env = IsaacLabVecEnv(isaac_env.unwrapped)

    return vec_env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step, config)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    train_env = make_isaac_env(config)
    # reuse same env for eval (IsaacLab can only have one sim instance)
    eval_env = train_env

    acts = train_env.action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config.envs, 1).to(config.device),
                torch.tensor(acts.high).repeat(config.envs, 1).to(config.device),
            ),
            1,
        )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate_vec(
            random_agent,
            train_env,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_env.observation_space,
        train_env.action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    try:
        while agent._step < config.steps + config.eval_every:
            logger.write()
            if config.eval_episode_num > 0:
                print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate_vec(
                    eval_policy,
                    eval_env,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", tools.to_np(video_pred))
                state = None
            print("Start training.")
            state = tools.simulate_vec(
                agent,
                train_env,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, logdir / "latest.pt")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        logger.close(exit_code=1)
        train_env.close()
        return

    logger.close()
    train_env.close()


if __name__ == "__main__":
    # load dreamer configs
    _yaml = yaml.YAML(typ="safe", pure=True)
    configs = _yaml.load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args_cli.configs] if args_cli.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser2 = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser2.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser2.parse_args(remaining)

    main(config)
    simulation_app.close()

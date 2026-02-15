"""Train DreamerV3 with IsaacLab environments.

Launch IsaacSim before any IsaacLab imports, then run the dreamer training
loop using a vectorized IsaacLab environment.

Usage:
  python dreamer_isaaclab.py --configs isaac_proprio --task isaac_cartpole_balance --logdir ./logdir/isaac_proprio/isaac_cartpole_balance

  And to see the progress with TensorBoard:
  tensorboard --logdir ./logdir/isaac_proprio/isaac_cartpole_balance/
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
from dreamer import Dreamer, count_steps, make_dataset

# task name -> (gym_id, obs_names_dict_or_None)
ISAAC_TASKS = {
    "cartpole_balance": {
        "gym_id": "Isaac-Cartpole-Direct-v0",
        "obs_names": {"pole_pos": 1, "pole_vel": 1, "cart_pos": 1, "cart_vel": 1},
        "image_key": None,
        "reward_fn": "dmc_balance",  # use DMC-compatible reward for comparability
        "disable_termination": True,  # time-only truncation, like DMC
        "obs_transform": "dmc_cartpole",  # remap obs to DMC format (cos/sin angle)
    },
    "cartpole_balance_rgb": {
        "gym_id": "Isaac-Cartpole-RGB-Camera-Direct-v0",
        "obs_names": {"pole_pos": 1, "pole_vel": 1, "cart_pos": 1, "cart_vel": 1},
        "image_key": "image",
        "reward_fn": "dmc_balance",
        "disable_termination": True,
        "obs_transform": "dmc_cartpole",
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
    obs_names = task_info["obs_names"]
    image_key = task_info["image_key"]
    reward_fn = task_info.get("reward_fn", None)
    disable_termination = task_info.get("disable_termination", False)
    obs_transform = task_info.get("obs_transform", None)

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
        # match DMC: render at the exact target resolution so no resize is
        # needed (DMC renders directly into a 64x64 framebuffer).
        env_cfg.tiled_camera.width = config.size[1]
        env_cfg.tiled_camera.height = config.size[0]
        # Disable anti-aliasing to match DMC's raw OpenGL rasteriser.
        # IsaacLab defaults to DLSS which smooths the image significantly.
        from isaaclab.sim import RenderCfg

        env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")
        if task == "cartpole_balance_rgb":
            # DMC's fixed camera: pos="0 -4 1" (4m back, cart height).
            # IsaacLab cart is at z=2.0, camera looks along +X, so the
            # equivalent is (-4, 0, 2).  Default was (-5, 0, 2).
            env_cfg.tiled_camera.offset.pos = (-4.0, 0.0, 2.0)
    else:
        render_mode = None

    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
    vec_env = IsaacLabVecEnv(
        isaac_env.unwrapped,
        obs_names=obs_names,
        image_key=image_key,
        size=tuple(config.size),
        reward_fn=reward_fn,
        action_repeat=config.action_repeat,
        disable_termination=disable_termination,
        obs_transform=obs_transform,
    )

    # For the vision cartpole task, override scene colours to match DMC.
    # This must happen after gym.make (scene exists) but before training
    # starts collecting frames.
    if task == "cartpole_balance_rgb":
        IsaacLabVecEnv.apply_dmc_cartpole_colors(isaac_env.unwrapped)

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

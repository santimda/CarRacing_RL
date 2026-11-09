"""Training, evaluation, checkpoint, and reward utilities for CarRacing."""

import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import FloatSchedule
from gymnasium.wrappers import FrameStackObservation

from drq_sac import DrQSAC
from plot_tensorboard import plot_tensorboard_metrics

ROOT = Path(__file__).resolve().parent
ENV_ID = "CarRacing-v3"
ALGORITHMS = {"PPO": PPO, "SAC": SAC, "DRQ-SAC": DrQSAC}
OFF_POLICY_ALGORITHMS = {"SAC", "DRQ-SAC"}


class ContinuationLinearSchedule:
    """Linear schedule normalized to one resumed learn() invocation."""

    def __init__(
        self, initial_value: float, final_value: float, initial_progress: float
    ):
        self.initial_value = float(initial_value)
        self.final_value = float(final_value)
        self.initial_progress = float(initial_progress)

    def __call__(self, progress_remaining: float) -> float:
        if self.initial_progress <= 0:
            return self.final_value
        fraction = min(
            max(float(progress_remaining) / self.initial_progress, 0.0), 1.0
        )
        return self.final_value + fraction * (self.initial_value - self.final_value)


class RewardWrapper(gym.Wrapper):
    """Apply the configured reward shaping to an environment."""

    def __init__(self, env, reward_config: dict):
        super().__init__(env)
        self.reward_config = reward_config
        self._previous_steering_sign = 0
        self._same_direction_steps = 0
        self._no_progress_steps = 0
        self._base_episode_return = 0.0
        self._shaped_episode_return = 0.0

    def reset(self, **kwargs):
        self._previous_steering_sign = 0
        self._same_direction_steps = 0
        self._no_progress_steps = 0
        self._base_episode_return = 0.0
        self._shaped_episode_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        car = getattr(self.unwrapped, "car", None)
        hull = getattr(car, "hull", None)
        velocity = getattr(hull, "linearVelocity", None)
        if velocity is not None and hull is not None:
            forward = hull.GetWorldVector((0, 1))
            forward_speed = float(forward[0] * velocity.x + forward[1] * velocity.y)
        else:
            forward_speed = 0.0

        steering, throttle, brake = np.asarray(action, dtype=np.float32)
        base_reward = float(reward)
        made_progress = base_reward > 0.0
        if made_progress:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        speed_scale = self.reward_config["speed_scale"]
        target_speed = self.reward_config["target_speed"]
        clipped_forward_speed = min(max(forward_speed, 0.0), target_speed)
        speed_bonus = (
            speed_scale * clipped_forward_speed if made_progress else 0.0
        )
        throttle_bonus = self.reward_config["throttle_bonus"] * float(throttle)
        brake_penalty = -self.reward_config["brake_penalty"] * float(brake)

        threshold = self.reward_config["steering_threshold"]
        steering_sign = int(np.sign(steering)) if abs(steering) >= threshold else 0
        if steering_sign == 0 or steering_sign != self._previous_steering_sign:
            self._same_direction_steps = 0
        else:
            self._same_direction_steps += 1
        self._previous_steering_sign = steering_sign
        persistence = max(
            0, self._same_direction_steps - self.reward_config["steering_grace_steps"]
        )
        persistence = min(
            persistence, self.reward_config["steering_penalty_cap_steps"]
        )
        steering_penalty = -self.reward_config["steering_penalty"] * persistence

        no_progress_persistence = max(
            0,
            self._no_progress_steps
            - self.reward_config["no_progress_grace_steps"],
        )
        no_progress_persistence = min(
            no_progress_persistence,
            self.reward_config["no_progress_penalty_cap_steps"],
        )
        no_progress_penalty = (
            -self.reward_config["no_progress_penalty"]
            * no_progress_persistence
        )
        lap_completion_bonus = (
            self.reward_config["lap_completion_bonus"]
            if info.get("lap_finished", False)
            else 0.0
        )
        shaped_reward = (
            base_reward
            + speed_bonus
            + throttle_bonus
            + brake_penalty
            + steering_penalty
            + no_progress_penalty
            + lap_completion_bonus
        )
        self._base_episode_return += base_reward
        self._shaped_episode_return += shaped_reward
        info = dict(info)
        info.update(
            {
                "base_reward": base_reward,
                "forward_speed": forward_speed,
                "speed_bonus": speed_bonus,
                "throttle_bonus": throttle_bonus,
                "brake_penalty": brake_penalty,
                "made_progress": made_progress,
                "no_progress_steps": self._no_progress_steps,
                "no_progress_penalty": no_progress_penalty,
                "steering_persistence_steps": self._same_direction_steps,
                "steering_persistence_penalty": steering_penalty,
                "lap_completion_bonus": lap_completion_bonus,
            }
        )
        if terminated or truncated:
            info["base_episode_return"] = self._base_episode_return
            info["shaped_episode_return"] = self._shaped_episode_return
        return observation, shaped_reward, terminated, truncated, info


class ChannelFirstFrameStack(gym.ObservationWrapper):
    """Convert stacked HWC frames into the channel-first SB3 image format."""

    def __init__(self, env):
        super().__init__(env)
        stack_size, height, width, channels = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_size * channels, height, width),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        stack_size, height, width, channels = observation.shape
        return np.transpose(observation, (0, 3, 1, 2)).reshape(
            stack_size * channels, height, width
        )


def make_env(
    seed: int,
    algorithm: str = "PPO",
    reward_config: dict | None = None,
    render: bool = False,
    monitor: bool = False,
    observation_stack: int = 1,
    action_mode: str = "three_action",
) -> gym.Env:
    """Create an environment with configured observations and actions."""
    render_mode = "rgb_array" if render else None
    env = gym.make(ENV_ID, continuous=True, render_mode=render_mode)
    if algorithm == "DRQ-SAC":
        if reward_config is None:
            raise ValueError("DRQ-SAC requires reward_config")
        env = RewardWrapper(env, reward_config)
        if action_mode != "three_action":
            raise ValueError("DRQ-SAC requires the native three_action control space")
    elif algorithm == "SAC":
        if action_mode != "three_action":
            raise ValueError("SAC requires the native three_action control space")
    elif algorithm != "PPO":
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if algorithm in {"SAC", "DRQ-SAC"}:
        if observation_stack < 1:
            raise ValueError("observation_stack must be positive")
        if observation_stack > 1:
            env = FrameStackObservation(env, stack_size=observation_stack)
            env = ChannelFirstFrameStack(env)
    if monitor:
        env = Monitor(env)
    env.action_space.seed(seed)
    return env


def inspect_environment(seed: int) -> dict:
    env = make_env(seed)
    try:
        observation, reset_info = env.reset(seed=seed)
        action = env.action_space.sample()
        _, reward, terminated, truncated, step_info = env.step(action)
        return {
            "environment": ENV_ID,
            "gymnasium_version": gym.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "device": "cpu" if not torch.cuda.is_available() else "cuda available",
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "sample_action": np.asarray(action).tolist(),
            "reward_after_one_step": float(reward),
            "step_terminated": bool(terminated),
            "step_truncated": bool(truncated),
            "reset_info_keys": sorted(reset_info.keys()),
            "step_info_keys": sorted(step_info.keys()),
        }
    finally:
        env.close()


def run_episode(
    env: gym.Env,
    seed: int,
    policy=None,
    deterministic: bool = True,
    gif_path: Path | None = None,
    frame_stride: int = 4,
) -> dict:
    observation, _ = env.reset(seed=seed)
    shaped_return = 0.0
    base_return = 0.0
    length = 0
    final_info = {}
    terminated = truncated = False
    frames = []
    while not (terminated or truncated):
        if gif_path is not None and length % frame_stride == 0:
            frames.append(env.render())
        action = (
            env.action_space.sample()
            if policy is None
            else policy.predict(observation, deterministic=deterministic)[0]
        )
        observation, reward, terminated, truncated, final_info = env.step(action)
        shaped_return += float(reward)
        base_return += float(final_info.get("base_reward", reward))
        length += 1
    if gif_path is not None:
        frames.append(env.render())
        imageio.mimsave(gif_path, frames, duration=0.05, loop=0)
    return {
        "seed": seed,
        "return": base_return,
        "base_return": base_return,
        "shaped_return": shaped_return,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
        "lap_finished": bool(final_info.get("lap_finished", False)),
        "final_info_keys": sorted(final_info.keys()),
    }


def build_model(algorithm: str, env: gym.Env, seed: int, settings: dict):
    model_settings = dict(settings)
    policy = model_settings.pop("policy", "CnnPolicy")
    device = model_settings.pop("device", "cpu")
    return ALGORITHMS[algorithm](
        policy,
        env,
        seed=seed,
        device=device,
        verbose=0,
        **model_settings,
    )


def _load_model_kwargs(
    algorithm: str,
    settings: dict | None = None,
    placeholder_buffer: bool = False,
) -> dict:
    """Provide compatibility values before SB3 reconstructs a saved model."""
    kwargs = {}
    if algorithm == "PPO":
        learning_rate = float((settings or {}).get("learning_rate", 3e-4))
        kwargs["custom_objects"] = {
            "learning_rate": learning_rate,
            "lr_schedule": FloatSchedule(learning_rate),
        }
    if placeholder_buffer and algorithm in OFF_POLICY_ALGORITHMS:
        kwargs["buffer_size"] = 1
    return kwargs


def _configure_ppo_resume(model: PPO, settings: dict, timesteps: int) -> dict:
    """Apply conservative fine-tuning settings to a loaded PPO model."""
    initial_learning_rate = float(settings.get("learning_rate_initial", 1e-4))
    final_learning_rate = float(settings.get("learning_rate_final", 2e-5))
    clip_range = float(settings.get("clip_range", 0.1))
    target_kl = float(settings.get("target_kl", 0.02))
    max_log_std = float(settings.get("max_log_std", -0.7))
    reset_log_std_optimizer_state = bool(
        settings.get("reset_log_std_optimizer_state", True)
    )
    if initial_learning_rate <= 0 or final_learning_rate <= 0:
        raise ValueError("PPO resume learning rates must be positive")
    if final_learning_rate > initial_learning_rate:
        raise ValueError("PPO resume final learning rate cannot exceed its initial value")
    if not 0 < clip_range < 1:
        raise ValueError("PPO resume clip_range must be between zero and one")
    if target_kl <= 0:
        raise ValueError("PPO resume target_kl must be positive")

    start_timesteps = int(model.num_timesteps)
    initial_progress = timesteps / (start_timesteps + timesteps)
    schedule = ContinuationLinearSchedule(
        initial_learning_rate, final_learning_rate, initial_progress
    )
    model.learning_rate = schedule
    model.lr_schedule = schedule
    model.clip_range = FloatSchedule(clip_range)
    model.target_kl = target_kl

    with torch.no_grad():
        std_before = model.policy.log_std.exp().detach().cpu().tolist()
        model.policy.log_std.clamp_(max=max_log_std)
        std_after = model.policy.log_std.exp().detach().cpu().tolist()
    optimizer_state_reset = False
    if reset_log_std_optimizer_state:
        optimizer_state_reset = (
            model.policy.optimizer.state.pop(model.policy.log_std, None) is not None
        )

    return {
        "learning_rate_initial": initial_learning_rate,
        "learning_rate_final": final_learning_rate,
        "clip_range": clip_range,
        "target_kl": target_kl,
        "max_log_std": max_log_std,
        "std_before": std_before,
        "std_after": std_after,
        "log_std_optimizer_state_reset": optimizer_state_reset,
        "schedule_start_timesteps": start_timesteps,
        "schedule_requested_timesteps": timesteps,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProgressCheckpointCallback(BaseCallback):
    def __init__(
        self,
        output_dir: Path,
        start_timesteps: int,
        requested_timesteps: int,
        algorithm: str,
        validation_seeds: list[int] | None = None,
        validation_interval: int | None = None,
        reward_config: dict | None = None,
        observation_stack: int = 1,
        action_mode: str = "three_action",
    ):
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.start_timesteps = start_timesteps
        self.requested_timesteps = requested_timesteps
        self.algorithm = algorithm
        self.validation_seeds = validation_seeds or []
        self.validation_interval = validation_interval
        self.reward_config = reward_config
        self.observation_stack = observation_stack
        self.action_mode = action_mode
        self.next_report = min(512, requested_timesteps)
        self.next_checkpoint = 8192
        self.next_validation = validation_interval
        self.started_at = 0.0

    def _on_training_start(self):
        self.started_at = time.perf_counter()

    def _on_step(self):
        episode_infos = [
            info for info in self.locals.get("infos", []) if "base_episode_return" in info
        ]
        if episode_infos:
            self.logger.record(
                "rollout/base_ep_rew_mean",
                float(np.mean([info["base_episode_return"] for info in episode_infos])),
            )
            self.logger.record(
                "rollout/shaped_ep_rew_mean",
                float(np.mean([info["shaped_episode_return"] for info in episode_infos])),
            )
        completed = self.num_timesteps - self.start_timesteps
        if completed >= self.next_report or completed >= self.requested_timesteps:
            print(
                f"{self.model.__class__.__name__}: {completed:,}/{self.requested_timesteps:,} "
                f"additional steps ({100 * completed / self.requested_timesteps:.0f}%), "
                f"elapsed {(time.perf_counter() - self.started_at) / 60:.1f} min",
                flush=True,
            )
            self.next_report += 512
        if completed >= self.next_checkpoint:
            self._save_checkpoint()
            self.next_checkpoint += 8192
        if self.next_validation is not None and completed >= self.next_validation:
            self._run_validation()
            self.next_validation += self.validation_interval
        return True

    def _on_rollout_end(self):
        if isinstance(self.model, PPO):
            standard_deviations = self.model.policy.log_std.detach().exp().cpu().tolist()
            for name, value in zip(("steering", "gas", "brake"), standard_deviations):
                self.logger.record(f"diagnostics/action_std_{name}", float(value))

    def _on_training_end(self):
        self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint = self.output_dir / "latest_checkpoint"
        temporary = self.output_dir / "checkpoint_tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        self.model.save(temporary / "model")
        if isinstance(self.model, SAC):
            self.model.save_replay_buffer(temporary / "replay_buffer.pkl")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        os.replace(temporary, checkpoint)
        print(f"Checkpoint saved at {self.num_timesteps:,} total steps.", flush=True)

    def _run_validation(self):
        episodes = []
        for seed in self.validation_seeds:
            env = make_env(
                seed=seed,
                algorithm=self.algorithm,
                reward_config=self.reward_config,
                observation_stack=self.observation_stack,
                action_mode=self.action_mode,
            )
            try:
                episodes.append(run_episode(env, seed, self.model, deterministic=True))
            finally:
                env.close()
        returns = [episode["base_return"] for episode in episodes]
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        self.logger.record("eval/mean_return", mean_return)
        self.logger.record("eval/std_return", std_return)
        print(
            f"Validation at {self.num_timesteps:,} steps: "
            f"{mean_return:.2f} +/- {std_return:.2f}",
            flush=True,
        )

        metadata_path = self.output_dir / "best_checkpoint.json"
        previous = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        )
        if previous is not None and mean_return <= previous["validation_mean_return"]:
            return
        best_path = self.output_dir / "best_model.zip"
        self.model.save(best_path)
        save_json(
            metadata_path,
            {
                "algorithm": self.algorithm,
                "actual_timesteps": int(self.model.num_timesteps),
                "validation_mean_return": mean_return,
                "validation_std_return": std_return,
                "validation_shaped_mean_return": float(
                    np.mean([episode["shaped_return"] for episode in episodes])
                ),
                "validation_seeds": self.validation_seeds,
                "model_path": str(best_path.relative_to(ROOT)),
                "selection_metric": "mean deterministic original CarRacing return",
                "observation_stack": self.observation_stack,
                "action_mode": self.action_mode,
            },
        )
        print(f"New best validation checkpoint: {mean_return:.2f}", flush=True)


def _load_checkpoint(algorithm: str, output_dir: Path, env, settings: dict | None = None):
    checkpoint = output_dir / "latest_checkpoint"
    checkpoint_model = checkpoint / "model.zip"
    model_path = output_dir / "model.zip"
    source = checkpoint_model if checkpoint_model.exists() else model_path
    if not source.exists():
        return None, 0
    replay_path = checkpoint / "replay_buffer.pkl"
    model = ALGORITHMS[algorithm].load(
        source,
        env=env,
        device="cpu",
        **_load_model_kwargs(
            algorithm,
            settings=settings,
            placeholder_buffer=algorithm in OFF_POLICY_ALGORITHMS and replay_path.exists(),
        ),
    )
    if algorithm in OFF_POLICY_ALGORITHMS and replay_path.exists():
        model.load_replay_buffer(replay_path)
        model.buffer_size = model.replay_buffer.buffer_size
    if algorithm in OFF_POLICY_ALGORITHMS:
        model.train_freq = TrainFreq(int(model.train_freq.frequency), TrainFrequencyUnit.STEP)
    return model, int(model.num_timesteps)


def evaluate_policy(
    algorithm: str,
    model_path: Path,
    seeds: list[int],
    reward_config: dict | None,
    render_seed: int | None = None,
    gif_path: Path | None = None,
    observation_stack: int = 1,
    action_mode: str = "three_action",
) -> dict:
    model = ALGORITHMS[algorithm].load(
        model_path,
        device="cpu",
        **_load_model_kwargs(
            algorithm,
            placeholder_buffer=algorithm in OFF_POLICY_ALGORITHMS,
        ),
    )
    episodes = []
    for index, seed in enumerate(seeds, 1):
        print(f"Evaluating {algorithm}: {index}/{len(seeds)} (seed {seed})", flush=True)
        env = make_env(
            algorithm=algorithm,
            seed=seed,
            reward_config=reward_config,
            render=seed == render_seed,
            observation_stack=observation_stack,
            action_mode=action_mode,
        )
        try:
            episodes.append(
                run_episode(env, seed, model, deterministic=True,
                            gif_path=gif_path if seed == render_seed else None)
            )
        finally:
            env.close()
    returns = [episode["base_return"] for episode in episodes]
    shaped_returns = [episode["shaped_return"] for episode in episodes]
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "mean_shaped_return": float(np.mean(shaped_returns)),
        "std_shaped_return": float(np.std(shaped_returns, ddof=1)) if len(shaped_returns) > 1 else 0.0,
        "actual_timesteps": int(model.num_timesteps),
        "model_sha256": file_sha256(model_path),
    }


def _render_best_gifs(
    algorithm,
    output_dir,
    model_path,
    episodes,
    demo_seed,
    reward_config,
    observation_stack=1,
    action_mode="three_action",
):
    demo_path = output_dir / "best_validation_checkpoint.gif"
    demo = evaluate_policy(
        algorithm,
        model_path,
        [demo_seed],
        reward_config,
        demo_seed,
        demo_path,
        observation_stack,
        action_mode,
    )
    best_episode = max(episodes, key=lambda episode: episode["base_return"])
    best_path = output_dir / "best_validation_episode.gif"
    selected = evaluate_policy(
        algorithm,
        model_path,
        [best_episode["seed"]],
        reward_config,
        best_episode["seed"],
        best_path,
        observation_stack,
        action_mode,
    )
    return {
        "gif_path": str(demo_path.relative_to(ROOT)),
        "gif_seed": demo_seed,
        "gif_episode_return": demo["episodes"][0]["base_return"],
        "best_episode_seed": best_episode["seed"],
        "best_episode_return": best_episode["base_return"],
        "best_episode_gif_path": str(best_path.relative_to(ROOT)),
        "best_episode_gif_return": selected["episodes"][0]["base_return"],
    }


def train_and_evaluate(
    algorithm: str,
    timesteps: int,
    training_seed: int,
    validation_seeds: list[int],
    output_dir: Path,
    settings: dict,
    reward_config: dict | None,
    demo_seed: int = 48,
    observation_stack: int = 1,
    action_mode: str = "three_action",
    resume: bool = False,
    resume_settings: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        training_seed,
        algorithm,
        reward_config,
        monitor=True,
        observation_stack=observation_stack,
        action_mode=action_mode,
    )
    start = time.perf_counter()
    model, start_timesteps = (
        _load_checkpoint(algorithm, output_dir, env, settings=settings)
        if resume
        else (None, 0)
    )
    if resume and model is None:
        env.close()
        raise FileNotFoundError(
            f"Cannot resume {algorithm}; no checkpoint found under {output_dir}"
        )
    resumed = model is not None
    resume_adjustments = None
    fresh_replay_warmup = 0
    if model is None:
        model = build_model(algorithm, env, training_seed, settings)
    elif algorithm == "PPO" and resume_settings is not None:
        checkpoint_model = output_dir / "latest_checkpoint" / "model.zip"
        fallback_model = output_dir / "model.zip"
        checkpoint_source = (
            checkpoint_model if checkpoint_model.exists() else fallback_model
        )
        resume_adjustments = _configure_ppo_resume(model, resume_settings, timesteps)
        resume_adjustments.update(
            {
                "source_model_path": str(checkpoint_source.relative_to(ROOT)),
                "source_model_sha256": file_sha256(checkpoint_source),
                "effective_n_steps": int(model.n_steps),
                "effective_batch_size": int(model.batch_size),
                "effective_n_epochs": int(model.n_epochs),
            }
        )
        print(
            "Applied PPO resume settings: "
            f"lr {resume_adjustments['learning_rate_initial']:.1e} -> "
            f"{resume_adjustments['learning_rate_final']:.1e}, "
            f"clip_range={resume_adjustments['clip_range']}, "
            f"target_kl={resume_adjustments['target_kl']}, "
            f"std {resume_adjustments['std_before']} -> "
            f"{resume_adjustments['std_after']}",
            flush=True,
        )
    elif algorithm in OFF_POLICY_ALGORITHMS:
        replay_path = output_dir / "latest_checkpoint" / "replay_buffer.pkl"
        if not replay_path.exists():
            fresh_replay_warmup = int(settings.get("learning_starts", 8192))
            model.learning_starts = model.num_timesteps + fresh_replay_warmup
            print(
                f"No replay buffer found; collecting {fresh_replay_warmup:,} "
                "fresh transitions before SAC updates.",
                flush=True,
            )
    if resumed and isinstance(model, DrQSAC) and "exploration_probability" in settings:
        model.exploration_probability = float(settings["exploration_probability"])
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(tensorboard_dir), ["tensorboard"]))
    validation_interval = None
    if resumed and algorithm == "PPO" and resume_settings is not None:
        validation_interval = int(resume_settings.get("validation_interval", 20480))
        if validation_interval <= 0:
            raise ValueError("PPO resume validation_interval must be positive")
    callback = ProgressCheckpointCallback(
        output_dir,
        start_timesteps,
        timesteps,
        algorithm,
        validation_seeds=validation_seeds if validation_interval is not None else None,
        validation_interval=validation_interval,
        reward_config=reward_config,
        observation_stack=observation_stack,
        action_mode=action_mode,
    )
    try:
        model.learn(
            total_timesteps=timesteps,
            log_interval=1,
            reset_num_timesteps=not resumed,
            progress_bar=False,
            callback=callback,
        )
        model.logger.dump(step=int(model.num_timesteps))
        model.save(output_dir / "model")
    finally:
        env.close()

    plot_path, tags = plot_tensorboard_metrics(
        tensorboard_dir, output_dir / "plots" / "training_metrics.png", algorithm
    )
    model_path = output_dir / "model.zip"
    evaluation = evaluate_policy(
        algorithm,
        model_path,
        validation_seeds,
        reward_config,
        observation_stack=observation_stack,
        action_mode=action_mode,
    )
    latest_gif_path = output_dir / "latest_checkpoint.gif"
    latest_demo = evaluate_policy(
        algorithm,
        model_path,
        [demo_seed],
        reward_config,
        render_seed=demo_seed,
        gif_path=latest_gif_path,
        observation_stack=observation_stack,
        action_mode=action_mode,
    )
    evaluation["algorithm"] = algorithm
    evaluation["settings"] = settings
    evaluation["reward_config"] = reward_config
    evaluation["observation_stack"] = observation_stack
    evaluation["action_mode"] = action_mode
    evaluation["starting_timesteps"] = start_timesteps
    evaluation["incremental_timesteps"] = int(model.num_timesteps) - start_timesteps
    evaluation["fresh_replay_warmup_steps"] = fresh_replay_warmup
    evaluation["resume_adjustments"] = resume_adjustments
    evaluation["training_seconds"] = time.perf_counter() - start
    evaluation["tensorboard_plot"] = str(plot_path.relative_to(ROOT))
    evaluation["tensorboard_tags"] = tags
    evaluation["latest_gif_path"] = str(latest_gif_path.relative_to(ROOT))
    evaluation["latest_gif_seed"] = demo_seed
    evaluation["latest_gif_return"] = latest_demo["episodes"][0]["base_return"]
    evaluation["latest_gif_shaped_return"] = latest_demo["episodes"][0]["shaped_return"]
    return evaluation


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

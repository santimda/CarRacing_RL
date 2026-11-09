"""Independent PPO, SAC, and DrQ-SAC training commands for CarRacing."""

import argparse
import json
import shutil
from pathlib import Path

from experiment import (
    ALGORITHMS,
    OFF_POLICY_ALGORITHMS,
    ROOT,
    evaluate_policy,
    inspect_environment,
    save_json,
    train_and_evaluate,
)

CONFIG_PATH = ROOT / "experiment_config.json"
OUTPUT_ROOT = ROOT / "outputs"


def output_dir(algorithm: str) -> Path:
    return OUTPUT_ROOT / {
        "PPO": "ppo",
        "SAC": "sac",
        "DRQ-SAC": "drq_sac",
    }[algorithm]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def algorithm_environment_options(config: dict, algorithm: str) -> dict:
    algorithm_config = config["algorithms"][algorithm]
    return {
        "reward_config": algorithm_config.get("reward_shaping"),
        "observation_stack": algorithm_config.get("observation_stack", 1),
        "action_mode": algorithm_config.get("action_mode", "three_action"),
    }


def run_training(config: dict, algorithm: str, timesteps: int, resume: bool = False) -> dict:
    algorithm_config = config["algorithms"][algorithm]
    settings = algorithm_config["settings"]
    environment_options = algorithm_environment_options(config, algorithm)
    resume_settings = algorithm_config.get("resume_settings")
    directory = output_dir(algorithm)
    if not resume and directory.exists():
        shutil.rmtree(directory)
    result_path = directory / "results.json"
    previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    result = train_and_evaluate(
        algorithm=algorithm,
        timesteps=timesteps,
        training_seed=config["training_seed"],
        validation_seeds=config["validation_seeds"],
        output_dir=directory,
        settings=settings,
        demo_seed=config["demo_seed"],
        resume=resume,
        resume_settings=resume_settings,
        **environment_options,
    )
    history = list(previous.get("training_history", []))
    history.append(
        {
            "starting_timesteps": result["starting_timesteps"],
            "incremental_timesteps": result["incremental_timesteps"],
            "actual_timesteps": result["actual_timesteps"],
            "training_seconds": result["training_seconds"],
        }
    )
    result["training_history"] = history
    result["best_checkpoint"] = update_best_checkpoint(
        algorithm, directory, result, config
    )
    save_json(result_path, result)
    print(f"Results saved to {result_path.relative_to(ROOT)}")
    return result


def update_best_checkpoint(algorithm: str, directory: Path, result: dict, config: dict) -> dict:
    best_path = directory / "best_model.zip"
    metadata_path = directory / "best_checkpoint.json"
    current_path = directory / "model.zip"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    if previous is None or result["mean_return"] > previous["validation_mean_return"]:
        shutil.copy2(current_path, best_path)
        best = {
            "algorithm": algorithm,
            "actual_timesteps": result["actual_timesteps"],
            "validation_mean_return": result["mean_return"],
            "validation_std_return": result["std_return"],
            "validation_shaped_mean_return": result["mean_shaped_return"],
            "validation_seeds": config["validation_seeds"],
            "model_path": str(best_path.relative_to(ROOT)),
            "selection_metric": "mean deterministic original CarRacing return",
            "observation_stack": result.get("observation_stack", 1),
            "action_mode": result.get("action_mode", "three_action"),
        }
        gif_info = _render_best_gifs_for_result(algorithm, directory, best_path, result, config)
        best.update(gif_info)
        save_json(metadata_path, best)
        return best
    return previous


def restore_checkpoint(algorithm: str, checkpoint: str) -> None:
    directory = output_dir(algorithm)
    source = directory / ("best_model.zip" if checkpoint == "best" else "model.zip")
    if not source.exists():
        raise FileNotFoundError(f"Missing {checkpoint} checkpoint: {source}")
    shutil.copy2(source, directory / "model.zip")
    latest = directory / "latest_checkpoint"
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, latest / "model.zip")
    if algorithm in OFF_POLICY_ALGORITHMS:
        replay = latest / "replay_buffer.pkl"
        if checkpoint == "best":
            # The best model has no replay copy. Avoid pairing it with the
            # latest model's replay; training will collect a fresh warmup.
            if replay.exists():
                replay.unlink()
        elif not replay.exists():
            raise FileNotFoundError(
                f"Missing latest replay buffer: {replay}"
            )
    print(f"Restored {algorithm} {checkpoint} checkpoint to {directory.relative_to(ROOT)}")


def _render_best_gifs_for_result(algorithm, directory, model_path, result, config):
    from experiment import _render_best_gifs

    return _render_best_gifs(
        algorithm,
        directory,
        model_path,
        result["episodes"],
        config["demo_seed"],
        **algorithm_environment_options(config, algorithm),
    )


def run_evaluation(config: dict, algorithm: str, checkpoint: str) -> dict:
    directory = output_dir(algorithm)
    model_path = directory / ("best_model.zip" if checkpoint == "best" else "model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {checkpoint} checkpoint: {model_path}")
    result = evaluate_policy(
        algorithm,
        model_path,
        config["validation_seeds"],
        **algorithm_environment_options(config, algorithm),
    )
    print(
        f"{algorithm} original return: {result['mean_return']:.2f} ± "
        f"{result['std_return']:.2f}; shaped return: {result['mean_shaped_return']:.2f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "evaluate", "restore", "smoke"))
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="PPO")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--checkpoint", choices=("latest", "best"), default="latest")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the algorithm's latest checkpoint; training is fresh by default",
    )
    args = parser.parse_args()
    config = load_config()
    if args.timesteps is not None and args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.resume and args.command != "train":
        parser.error("--resume is only valid with the train command")

    if args.command == "check":
        print(json.dumps(inspect_environment(config["training_seed"]), indent=2))
    elif args.command == "train":
        timesteps = args.timesteps or config["default_increment"]
        run_training(config, args.algorithm, timesteps, resume=args.resume)
    elif args.command == "evaluate":
        run_evaluation(config, args.algorithm, args.checkpoint)
    elif args.command == "restore":
        restore_checkpoint(args.algorithm, args.checkpoint)
    else:
        smoke_steps = args.timesteps or config["smoke_timesteps"]
        run_training(config, args.algorithm, smoke_steps)


if __name__ == "__main__":
    main()

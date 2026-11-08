"""Export selected Stable-Baselines3 metrics as PNG plots."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PANEL_CONFIG = {
    "PPO": {
        "episode_return": {
            "show": True,
            "tag": "rollout/ep_rew_mean",
            "title": "Episode return",
        },
        "approx_kl": {
            "show": True,
            "tag": "train/approx_kl",
            "title": "Approximate KL",
        },
        "clip_fraction": {
            "show": True,
            "tag": "train/clip_fraction",
            "title": "Clip fraction",
        },
        "explained_variance": {
            "show": True,
            "tag": "train/explained_variance",
            "title": "Explained variance",
        },
        "episode_length": {
            "show": False,
            "tag": "rollout/ep_len_mean",
            "title": "Episode length",
        },
        "entropy_loss": {
            "show": False,
            "tag": "train/entropy_loss",
            "title": "Entropy loss",
        },
        "policy_gradient_loss": {
            "show": False,
            "tag": "train/policy_gradient_loss",
            "title": "Policy gradient loss",
        },
        "value_loss": {
            "show": False,
            "tag": "train/value_loss",
            "title": "Value loss",
        },
    },
    "SAC": {
        "episode_return": {
            "show": True,
            "tag": "rollout/ep_rew_mean",
            "title": "Episode return",
        },
        "actor_loss": {
            "show": True,
            "tag": "train/actor_loss",
            "title": "Actor loss",
        },
        "critic_loss": {
            "show": True,
            "tag": "train/critic_loss",
            "title": "Critic loss",
        },
        "episode_length": {
            "show": False,
            "tag": "rollout/ep_len_mean",
            "title": "Episode length",
        },
        "entropy_coefficient": {
            "show": False,
            "tag": "train/ent_coef",
            "title": "Entropy coefficient",
        },
        "learning_rate": {
            "show": False,
            "tag": "train/learning_rate",
            "title": "Learning rate",
        },
    },
    "DRQ-SAC": {
        "original_return": {
            "show": True,
            "tag": "rollout/base_ep_rew_mean",
            "title": "Original episode return",
        },
        "shaped_return": {
            "show": True,
            "tag": "rollout/shaped_ep_rew_mean",
            "title": "Shaped episode return",
        },
        "actor_loss": {
            "show": True,
            "tag": "train/actor_loss",
            "title": "Actor loss",
        },
        "critic_loss": {
            "show": True,
            "tag": "train/critic_loss",
            "title": "Critic loss",
        },
        "episode_length": {
            "show": False,
            "tag": "rollout/ep_len_mean",
            "title": "Episode length",
        },
        "entropy_coefficient": {
            "show": False,
            "tag": "train/ent_coef",
            "title": "Entropy coefficient",
        },
        "learning_rate": {
            "show": False,
            "tag": "train/learning_rate",
            "title": "Learning rate",
        },
    },
}


def plot_tensorboard_metrics(
    log_dir: Path, output_path: Path, algorithm: str
) -> tuple[Path, list[str]]:
    """Plot available metrics from one training run."""
    if algorithm not in PANEL_CONFIG:
        raise ValueError(f"No TensorBoard plot configuration for {algorithm}")

    accumulators = []
    for event_file in sorted(log_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        accumulators.append(accumulator)
    if not accumulators:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")

    series = {}
    selected_panels = [
        panel for panel in PANEL_CONFIG[algorithm].values() if panel["show"]
    ]
    if not selected_panels:
        raise ValueError(f"No panels enabled for {algorithm}")

    for panel in selected_panels:
        tag = panel["tag"]
        points = []
        for accumulator in accumulators:
            if tag in accumulator.Tags().get("scalars", []):
                points.extend(
                    (event.step, event.value) for event in accumulator.Scalars(tag)
                )
        if points:
            series[panel["title"]] = sorted(points, key=lambda point: point[0])
    if not series:
        raise ValueError(f"No configured scalar tags found for {algorithm} in {log_dir}")

    columns = 2
    rows = math.ceil(len(series) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.5 * rows), squeeze=False)
    for axis, (title, points) in zip(axes.flat, series.items()):
        steps, values = zip(*points)
        axis.plot(steps, values, linewidth=1.4, marker="o", markersize=3)
        axis.set_title(title)
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(series):]:
        axis.set_visible(False)

    figure.suptitle(f"{algorithm} training metrics", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    plotted_tags = [
        panel["tag"] for panel in selected_panels if panel["title"] in series
    ]
    return output_path, plotted_tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--algorithm", choices=PANEL_CONFIG, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path, tags = plot_tensorboard_metrics(
        args.log_dir, args.output, args.algorithm
    )
    print(f"Saved {output_path}; plotted tags: {', '.join(tags)}")


if __name__ == "__main__":
    main()

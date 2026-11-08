"""DrQ-style SAC with random-shift image augmentation."""

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


def random_shift(images: th.Tensor, pad: int = 4) -> th.Tensor:
    """Apply independent integer translations without looping over the batch."""
    if pad == 0:
        return images
    batch_size, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    offsets = th.randint(
        0, 2 * pad + 1, (batch_size, 2), device=images.device
    )
    windows = padded.unfold(2, height, 1).unfold(3, width, 1)
    batch_indices = th.arange(batch_size, device=images.device)
    return windows[batch_indices, :, offsets[:, 0], offsets[:, 1]]


class DrQSAC(SAC):
    """SAC with DrQ augmentation and a critic-trained shared image encoder."""

    def __init__(
        self,
        *args,
        augmentation_pad: int = 4,
        exploration_probability: float = 0.0,
        warmup_full_steering_probability: float = 0.2,
        **kwargs,
    ):
        self.augmentation_pad = augmentation_pad
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be between 0 and 1")
        if not 0.0 <= warmup_full_steering_probability <= 1.0:
            raise ValueError("warmup_full_steering_probability must be between 0 and 1")
        self.exploration_probability = exploration_probability
        self.warmup_full_steering_probability = warmup_full_steering_probability
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.policy.share_features_extractor:
            raise ValueError(
                "DrQSAC requires share_features_extractor=False before setup"
            )

        # The critic owns the encoder and trains it; the actor reuses its features.
        self.actor.features_extractor = self.critic.features_extractor
        actor_parameters = [
            parameter
            for name, parameter in self.actor.named_parameters()
            if "features_extractor" not in name
        ]
        self.actor.optimizer = self.policy.optimizer_class(
            actor_parameters,
            lr=self.lr_schedule(1),
            **self.policy.optimizer_kwargs,
        )

    @staticmethod
    def _set_requires_grad(module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def _structured_actions(self, n_envs: int) -> np.ndarray:
        """Sample safe warmup actions for the native three-action space."""
        full_steering = (
            np.random.random(n_envs) < self.warmup_full_steering_probability
        )
        steering = np.where(
            full_steering,
            np.random.uniform(-1.0, 1.0, size=n_envs),
            np.random.uniform(-0.3, 0.3, size=n_envs),
        )
        if self.action_space.shape[0] != 3:
            raise ValueError(f"Expected native three-action space, got {self.action_space.shape}")
        actions = np.zeros((n_envs, 3), dtype=np.float32)
        actions[:, 0] = steering
        throttle = np.random.random(n_envs) < 0.75
        actions[throttle, 1] = np.random.uniform(
            0.5, 1.0, size=throttle.sum()
        )
        actions[~throttle, 2] = np.random.uniform(
            0.5, 1.0, size=(~throttle).sum()
        )
        return actions

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """Use structured driving actions during warmup and intermittently after it."""
        if not isinstance(self.action_space, spaces.Box):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        use_structured = self.num_timesteps < learning_starts
        use_structured = use_structured or np.random.random() < self.exploration_probability
        if not use_structured:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        unscaled_action = self._structured_actions(n_envs)
        scaled_action = self.policy.scale_action(unscaled_action)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
        return self.policy.unscale_action(scaled_action), scaled_action

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            observations = random_shift(replay_data.observations, self.augmentation_pad)
            next_observations = random_shift(
                replay_data.next_observations, self.augmentation_pad
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
                next_q_values = th.cat(
                    self.critic_target(next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Recompute actor outputs after the critic step. The shared encoder
            # changed in-place during critic optimization.
            self._set_requires_grad(self.critic, False)
            try:
                actions_pi, log_prob = self.actor.action_log_prob(observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(
                        self.log_ent_coef
                        * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                ent_coefs.append(ent_coef.item())
                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            finally:
                self._set_requires_grad(self.critic, True)

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

# -*- coding: utf-8 -*-
"""
agents/ql_diffusion.py
----------------------
Diffusion Q-Learning (Diffusion_QL) for continuous autonomous driving control.

Combines a diffusion-based Actor with twin Q-network Critics.
Training balances behaviour cloning loss and Q-learning loss.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import logger
from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    """Twin Q-network critic (Q1 and Q2) to mitigate overestimation bias."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),             nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),             nn.Mish(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1_model = _mlp()
        self.q2_model = _mlp()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    """
    Diffusion Q-Learning agent.

    Args:
        state_dim    : dimensionality of the observation vector
        action_dim   : dimensionality of the action vector
        max_action   : action clipping bound (typically 1.0)
        device       : torch device
        discount     : RL discount factor γ
        tau          : soft-update rate for target networks
        max_q_backup : if True, use max-Q backup (10 samples per next state)
        eta          : weighting coefficient for Q-learning loss vs BC loss
        beta_schedule: noise schedule for diffusion ('linear' or 'vp')
        n_timesteps  : denoising steps in the diffusion model
        ema_decay    : EMA decay rate for actor target
        lr           : learning rate for both actor and critic
        lr_decay     : enable cosine LR annealing
        lr_maxt      : number of steps for one cosine cycle
        grad_norm    : max gradient norm for clipping (0 = disabled)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        max_q_backup=False,
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=1.0,
    ):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.lr_decay  = lr_decay
        self.grad_norm = grad_norm

        self.step            = 0
        self.step_start_ema  = step_start_ema
        self.ema             = EMA(ema_decay)
        self.ema_model       = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic        = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler  = CosineAnnealingLR(self.actor_optimizer,  T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim   = state_dim
        self.max_action  = max_action
        self.action_dim  = action_dim
        self.discount    = discount
        self.tau         = tau
        self.eta         = eta
        self.device      = device
        self.max_q_backup = max_q_backup

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        """Run `iterations` gradient updates from the replay buffer."""

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

        for _ in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # ── Critic update ──────────────────────────────────────────
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                # Sample 10 next actions, keep the max Q
                next_state_rpt  = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                tq1, tq2 = self.critic_target(next_state_rpt, next_action_rpt)
                tq1 = tq1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                tq2 = tq2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(tq1, tq2)
            else:
                next_action = self.ema_model(next_state)
                tq1, tq2    = self.critic_target(next_state, next_action)
                target_q    = torch.min(tq1, tq2)

            target_q    = (reward + not_done * self.discount * target_q).detach()
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            # ── Actor update ───────────────────────────────────────────
            bc_loss    = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new, q2_new = self.critic(state, new_action)
            # Randomly choose which Q to differentiate through (reduces variance)
            if np.random.uniform() > 0.5:
                q_loss = -q1_new.mean() / q2_new.abs().mean().detach()
            else:
                q_loss = -q2_new.mean() / q1_new.abs().mean().detach()

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            # ── EMA & soft target update ───────────────────────────────
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.step += 1

            # ── Logging ────────────────────────────────────────────────
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm',  actor_grad_norms.max().item(),  self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss',       bc_loss.item(),       self.step)
                log_writer.add_scalar('QL Loss',       q_loss.item(),        self.step)
                log_writer.add_scalar('Critic Loss',   critic_loss.item(),   self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample_action(self, state):
        """
        Sample an action for deployment.

        Generates 50 candidate actions via the diffusion policy and selects
        the one with the highest Q-value (softmax-weighted multinomial draw).
        """
        state     = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)

        with torch.no_grad():
            action  = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx     = torch.multinomial(F.softmax(q_value, dim=0), 1)

        return action[idx].cpu().data.numpy().flatten()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, dir, id=None):
        suffix = f'_{id}' if id is not None else ''
        torch.save(self.actor.state_dict(),  f'{dir}/actor{suffix}.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic{suffix}.pth')

    def load_model(self, dir, id=None):
        suffix = f'_{id}' if id is not None else ''
        self.actor.load_state_dict( torch.load(f'{dir}/actor{suffix}.pth',  map_location=self.device))
        self.critic.load_state_dict(torch.load(f'{dir}/critic{suffix}.pth', map_location=self.device))

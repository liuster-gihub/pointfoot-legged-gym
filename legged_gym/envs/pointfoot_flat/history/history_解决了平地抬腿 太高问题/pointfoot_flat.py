import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .pointfoot_flat_config import BipedCfgPF

class BipedPF(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                < self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        zero_command_env_ids = env_ids[zero_command_idx]
        self.commands[zero_command_env_ids, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_env_ids],
                self.forward_vec[zero_command_env_ids],
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_env_ids, 3] = heading

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[18:] = 0.0  # previous actions
        return noise_vec
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        lateral_error, heading_error = self._get_path_errors()
        adaptive_clearance = self._get_adaptive_clearance_peaks()
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
                lateral_error.unsqueeze(1),
                torch.sin(heading_error).unsqueeze(1),
                torch.cos(heading_error).unsqueeze(1),
                adaptive_clearance / self.cfg.rewards.feet_height_target,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    def _get_path_errors(self):
        """Return signed errors relative to the +x stair centerline."""
        lateral_error = self.base_position[:, 1] - self.env_origins[:, 1]
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading_error = wrap_to_pi(torch.atan2(forward[:, 1], forward[:, 0]))
        return lateral_error, heading_error
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_stair_progress(self):
        # Reward progress along the stair direction. Tracking reward still limits overspeed.
        forward_speed = (self.base_position[:, 0] - self.last_base_position[:, 0]) / self.dt
        return torch.clip(forward_speed, 0.0, self.cfg.commands.ranges.lin_vel_x[1])

    def _reward_lateral_motion(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_yaw_motion(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_lateral_position(self):
        lateral_error, _ = self._get_path_errors()
        return torch.square(lateral_error)

    def _reward_heading_error(self):
        _, heading_error = self._get_path_errors()
        return torch.square(heading_error)

    def _reward_swing_foot_clearance(self):
        swing_progress, in_swing = self._get_swing_progress()
        swing_mask = (1.0 - self.desired_contact_states) * in_swing.float()
        peak_height = self._get_adaptive_clearance_peaks()
        phase_profile = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        phase_profile = torch.pow(
            phase_profile, self.cfg.rewards.feet_height_phase_power
        )
        target_height = peak_height * phase_profile
        clearance = torch.exp(
            -torch.square(self.foot_heights - target_height)
            / self.cfg.rewards.feet_clearance_sigma
        )
        return torch.sum(swing_mask * clearance, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1e-6
        )

    def _get_swing_progress(self):
        """Return normalized [0, 1] swing progress and a per-foot swing mask."""
        offsets = self.gaits[:, 1]
        foot_phase = torch.remainder(
            torch.stack(
                (self.gait_indices, self.gait_indices + offsets), dim=1
            ),
            1.0,
        )
        stance_duration = self.gaits[:, 2].unsqueeze(1).expand_as(foot_phase)
        in_swing = foot_phase >= stance_duration
        swing_progress = torch.clamp(
            (foot_phase - stance_duration) / (1.0 - stance_duration + 1e-6),
            0.0,
            1.0,
        )
        return swing_progress, in_swing

    def _sample_terrain_heights_xy(self, points_xy):
        """Sample conservative terrain heights at world-frame XY points."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(points_xy.shape[:-1], device=self.device)
        if self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[..., 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[..., 1], 0, self.height_samples.shape[1] - 2)

        # Taking the maximum makes a stair edge visible before the foot reaches
        # the riser instead of averaging it away.
        heights = torch.maximum(
            self.height_samples[px, py], self.height_samples[px + 1, py]
        )
        heights = torch.maximum(heights, self.height_samples[px, py + 1])
        return heights * self.terrain.cfg.vertical_scale

    def _get_adaptive_clearance_peaks(self):
        """Compute each foot's peak clearance from terrain sampled ahead."""
        current_ground = self._sample_terrain_heights_xy(
            self.foot_positions[:, :, :2]
        )
        forward = quat_apply_yaw(self.base_quat, self.forward_vec)[:, :2]
        forward = forward / torch.norm(forward, dim=1, keepdim=True).clamp(min=1e-6)
        distances = torch.as_tensor(
            self.cfg.rewards.feet_height_lookahead,
            device=self.device,
            dtype=self.foot_positions.dtype,
        )
        sample_points = (
            self.foot_positions[:, :, None, :2]
            + forward[:, None, None, :] * distances[None, None, :, None]
        )
        upcoming_ground = self._sample_terrain_heights_xy(sample_points)
        obstacle_height = torch.clamp(
            torch.max(upcoming_ground, dim=-1).values - current_ground,
            min=0.0,
        )
        adaptive_peak = torch.maximum(
            torch.full_like(obstacle_height, self.cfg.rewards.feet_height_target_flat),
            obstacle_height + self.cfg.rewards.feet_height_safety_margin,
        )
        return torch.clamp(
            adaptive_peak, max=self.cfg.rewards.feet_height_target
        )

    def _reward_swing_foot_forward(self):
        swing_progress, in_swing = self._get_swing_progress()
        swing_mask = (1.0 - self.desired_contact_states) * in_swing.float()
        peak_height = self._get_adaptive_clearance_peaks()
        phase_profile = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        phase_profile = torch.pow(
            phase_profile, self.cfg.rewards.feet_height_phase_power
        )
        target_height = peak_height * phase_profile
        # Forward motion is rewarded only while the foot follows the requested
        # clearance trajectory; lifting above it no longer earns extra reward.
        height_gate = torch.exp(
            -torch.square(self.foot_heights - target_height)
            / self.cfg.rewards.feet_clearance_sigma
        )
        forward_vel = torch.clip(
            self.foot_relative_velocities[:, :, 0],
            0.0,
            self.cfg.rewards.swing_forward_vel_target,
        )
        return torch.sum(swing_mask * height_gate * forward_vel, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1e-6
        )

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

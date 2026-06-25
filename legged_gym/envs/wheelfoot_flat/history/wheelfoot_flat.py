import math
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BipedCfgWF, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(
            0, self.cfg.env.num_envs, device=self.device
        )

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def reset_idx(self, env_ids):
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
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.locomotion_mode[env_ids] = 0
        self.mode_steps[env_ids] = 0
        self.squeeze_confirm_steps[env_ids] = 0
        self.descent_confirm_steps[env_ids] = 0
        self.mode_clear_steps[env_ids] = 0
        leg_ids = [0, 1, 2, 4, 5, 6]
        self.kinematic_joint_ref[env_ids] = self.default_dof_pos[
            env_ids
        ][:, leg_ids]
        self.last_swing_mask[env_ids] = False
        self.trajectory_contact_hold[env_ids] = False
        self.wheel_blocking_forces[env_ids] = 0.0
        self.feedforward_amplitudes[env_ids] = 0.0
        self.feedforward_lift_state[env_ids] = 0.0
        self.filtered_blocking_forces[env_ids] = 0.0
        self.terrain_step_ahead[env_ids] = 0.0
        self.terrain_step_down_ahead[env_ids] = 0.0
        self.swing_step_target_height[env_ids] = 0.0
        self.current_feet_air_time[env_ids] = 0.0
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

    def step(self, actions):
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
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        leg_action_ids = [0, 1, 2, 4, 5, 6]
        residual_leg_actions = torch.cat(
            (actions[:, 0:3], actions[:, 4:7]), dim=1
        )
        desired_pos = self.default_dof_pos.clone()
        desired_pos[:, leg_action_ids] = (
            self.kinematic_joint_ref
            + self.cfg.control.residual_joint_scale * residual_leg_actions
        )
        desired_pos = torch.maximum(
            torch.minimum(desired_pos, self.dof_pos_limits[:, 1]),
            self.dof_pos_limits[:, 0],
        )

        wheel_radius = self.cfg.asset.foot_radius
        half_track = self.cfg.control.wheel_track_width * 0.5
        left_speed = (
            self.commands[:, 0] - half_track * self.commands[:, 2]
        ) / wheel_radius
        right_speed = (
            self.commands[:, 0] + half_track * self.commands[:, 2]
        ) / wheel_radius
        wheel_directions = torch.as_tensor(
            self.cfg.control.wheel_velocity_directions,
            device=self.device,
            dtype=self.dof_vel.dtype,
        )
        desired_wheel_speed = torch.stack(
            (left_speed, right_speed), dim=1
        ) * wheel_directions.unsqueeze(0)

        stepping = self.locomotion_mode == 1
        local_step = torch.max(self.terrain_step_ahead, dim=1).values
        local_step_down = torch.max(self.terrain_step_down_ahead, dim=1).values
        local_blocking = torch.max(self.filtered_blocking_forces, dim=1).values
        stair_control_active = stepping & (
            (local_step > self.cfg.kinematics.feedforward_step_low_m)
            | (local_blocking > self.cfg.kinematics.feedforward_force_low_n)
        ) & (
            local_step_down < self.cfg.kinematics.feedforward_step_low_m
        )
        if torch.any(stair_control_active):
            stance_mask = self.desired_contact_states > 0.5
            stair_speed = (
                self.cfg.control.stair_stance_wheel_speed
                * wheel_directions.unsqueeze(0)
            )
            desired_wheel_speed = torch.where(
                stair_control_active.unsqueeze(1),
                stance_mask.float() * stair_speed,
                desired_wheel_speed,
            )
        desired_wheel_speed += self.cfg.control.residual_wheel_speed_scale * \
            actions[:, [3, 7]]

        descent_mask = self.locomotion_mode == 2
        if torch.any(descent_mask):
            desired_wheel_speed[descent_mask] = torch.clamp(
                desired_wheel_speed[descent_mask],
                -self.cfg.control.descent_max_wheel_speed,
                self.cfg.control.descent_max_wheel_speed,
            )
        desired_vel = torch.zeros_like(self.dof_vel)
        desired_vel[:, [3, 7]] = desired_wheel_speed

        torques = self.p_gains * (desired_pos - self.dof_pos) + \
            self.d_gains * (desired_vel - self.dof_vel)
        if torch.any(descent_mask):
            leg_ids = [0, 1, 2, 4, 5, 6]
            torques[:, leg_ids] -= (
                descent_mask.float().unsqueeze(1)
                * self.cfg.control.descent_extra_leg_damping
                * self.dof_vel[:, leg_ids]
            )
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi
        lateral_error, heading_error = self._get_path_errors()
        mode_obs = torch.stack(
            (
                self.locomotion_mode == 0,
                self.locomotion_mode == 1,
                self.locomotion_mode == 2,
            ),
            dim=1,
        ).float()
        leg_tracking_error = (
            self.dof_pos[:, dof_list] - self.kinematic_joint_ref
        )
        wheel_contacts = (
            self.contact_forces[:, self.feet_indices, 2] > 0.1
        ).float()
        terrain_step_obs = torch.clamp(
            self.terrain_step_ahead
            / max(self.cfg.terrain.max_stair_step_height, 1e-6),
            0.0,
            2.0,
        )

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
                lateral_error.unsqueeze(1),
                torch.sin(heading_error).unsqueeze(1),
                torch.cos(heading_error).unsqueeze(1),
                mode_obs,
                self.kinematic_joint_ref,
                leg_tracking_error,
                wheel_contacts,
                terrain_step_obs,
                self.filtered_blocking_forces * self.obs_scales.contact_forces,
                self.feedforward_lift_state,
            ),
            dim=-1,
        )
        privileged_heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1)
            - self.cfg.rewards.base_height_target
            - self.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        critic_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                obs_buf,
                privileged_heights,
            ),
            dim=-1,
        )
        return obs_buf, critic_obs_buf

    def _get_path_errors(self):
        """Return signed errors relative to the +x stair centerline."""
        lateral_error = self.base_position[:, 1] - self.env_origins[:, 1]
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading_error = wrap_to_pi(torch.atan2(forward[:, 1], forward[:, 0]))
        return lateral_error, heading_error

    def _get_swing_progress(self):
        """Return normalized [0, 1] swing progress and a per-wheel swing mask."""
        offsets = self.gaits[:, 1]
        wheel_phase = torch.remainder(
            torch.stack(
                (self.gait_indices, self.gait_indices + offsets), dim=1
            ),
            1.0,
        )
        stance_duration = self.gaits[:, 2].unsqueeze(1).expand_as(wheel_phase)
        in_swing = wheel_phase >= stance_duration
        swing_progress = torch.clamp(
            (wheel_phase - stance_duration) / (1.0 - stance_duration + 1e-6),
            0.0,
            1.0,
        )
        return swing_progress, in_swing

    def _sample_terrain_heights_xy(self, points_xy):
        """Sample conservative terrain heights at world-frame XY points."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                points_xy.shape[:-1],
                device=self.device,
                dtype=points_xy.dtype,
            )
        if self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[..., 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[..., 1], 0, self.height_samples.shape[1] - 2)
        heights = torch.maximum(
            self.height_samples[px, py], self.height_samples[px + 1, py]
        )
        heights = torch.maximum(heights, self.height_samples[px, py + 1])
        return heights * self.terrain.cfg.vertical_scale

    def _update_terrain_step_ahead(self):
        """Estimate upcoming step height in front of each wheel for actor/controller."""
        if self.cfg.terrain.mesh_type in ["plane", "none"]:
            self.terrain_step_ahead[:] = 0.0
            self.terrain_step_down_ahead[:] = 0.0
            return

        forward = quat_apply(self.base_quat, self.forward_vec)
        forward_xy = forward[:, :2]
        forward_xy = forward_xy / (
            torch.norm(forward_xy, dim=1, keepdim=True) + 1e-6
        )
        lookahead = torch.as_tensor(
            self.cfg.rewards.feet_height_lookahead,
            device=self.device,
            dtype=self.foot_positions.dtype,
        )
        wheel_xy = self.foot_positions[:, :, :2]
        current_height = self._sample_terrain_heights_xy(wheel_xy)
        ahead_xy = (
            wheel_xy.unsqueeze(2)
            + forward_xy.unsqueeze(1).unsqueeze(2)
            * lookahead.view(1, 1, -1, 1)
        )
        ahead_height = self._sample_terrain_heights_xy(ahead_xy)
        max_ahead_height = torch.max(ahead_height, dim=2).values
        min_ahead_height = torch.min(ahead_height, dim=2).values
        step_height = max_ahead_height - current_height
        step_down_height = current_height - min_ahead_height
        self.terrain_step_ahead[:] = torch.clamp(
            step_height,
            min=0.0,
            max=self.cfg.terrain.max_stair_step_height,
        )
        self.terrain_step_down_ahead[:] = torch.clamp(
            step_down_height,
            min=0.0,
            max=self.cfg.terrain.max_stair_step_height,
        )

    def _get_adaptive_clearance_peaks(self):
        """Return clearance only for wheels that currently see an obstacle."""
        return self._get_stair_obstacle_gate(per_leg=True) * \
            self.cfg.rewards.feet_height_target

    def _get_stair_obstacle_gate(self, per_leg=False):
        """Gate stair swing logic by local step/blocked-wheel evidence."""
        mode_gate = (self.locomotion_mode == 1).float().unsqueeze(1)
        descent_edge = (
            self.terrain_step_down_ahead
            > self.cfg.kinematics.feedforward_step_low_m
        )
        obstacle = (
            (
                self.terrain_step_ahead
                > self.cfg.kinematics.feedforward_step_low_m
            )
            | (
                self.filtered_blocking_forces
                > self.cfg.kinematics.feedforward_force_low_n
            )
        ) & (~descent_edge)
        obstacle = obstacle.float()
        leg_gate = mode_gate * obstacle
        if per_leg:
            return leg_gate
        return (torch.max(leg_gate, dim=1).values > 0.5).float().unsqueeze(1)

    def _get_stair_mode_gate(self, adaptive_clearance=None):
        """Return a global per-env gate for squeeze-triggered stair stepping."""
        if adaptive_clearance is not None:
            return (adaptive_clearance > 1e-6).float()
        return (self.locomotion_mode == 1).float().unsqueeze(1)

    @staticmethod
    def _quintic_blend(value):
        value = torch.clamp(value, 0.0, 1.0)
        return value ** 3 * (10.0 - 15.0 * value + 6.0 * value ** 2)

    def _get_wheel_positions_base(self):
        wheel_positions_base = self.foot_positions - self.base_position.unsqueeze(1)
        for i in range(len(self.feet_indices)):
            wheel_positions_base[:, i, :] = quat_rotate_inverse(
                self.base_quat, wheel_positions_base[:, i, :]
            )
        return wheel_positions_base

    def _wheel_targets_to_joint_angles(self, wheel_targets_base):
        """Analytic 3-DoF leg IK for configurable abad/hip/knee geometry."""
        hip_offsets = torch.tensor(
            [
                [
                    self.cfg.kinematics.hip_offset_x,
                    self.cfg.kinematics.hip_offset_y,
                    self.cfg.kinematics.hip_offset_z,
                ],
                [
                    self.cfg.kinematics.hip_offset_x,
                    -self.cfg.kinematics.hip_offset_y,
                    self.cfg.kinematics.hip_offset_z,
                ],
            ],
            device=self.device,
            dtype=wheel_targets_base.dtype,
        )
        targets = wheel_targets_base - hip_offsets.unsqueeze(0)
        x = targets[:, :, 0]
        y = targets[:, :, 1]
        z = targets[:, :, 2]
        q_abad = torch.atan2(y, -z)
        down = torch.sqrt(torch.square(y) + torch.square(z) + 1e-8)

        l1 = self.cfg.kinematics.thigh_length
        l2 = self.cfg.kinematics.shank_length
        reach = torch.sqrt(torch.square(x) + torch.square(down))
        reach = torch.clamp(
            reach,
            min=abs(l1 - l2) + self.cfg.kinematics.ik_reach_margin,
            max=l1 + l2 - self.cfg.kinematics.ik_reach_margin,
        )
        scale = reach / torch.sqrt(torch.square(x) + torch.square(down) + 1e-8)
        x = x * scale
        down = down * scale
        cosine_knee = torch.clamp(
            (torch.square(x) + torch.square(down) - l1 ** 2 - l2 ** 2)
            / (2.0 * l1 * l2),
            -1.0,
            1.0,
        )
        q_knee = self.cfg.kinematics.knee_direction * torch.acos(cosine_knee)
        q_hip = torch.atan2(-x, down) - torch.atan2(
            l2 * torch.sin(q_knee), l1 + l2 * torch.cos(q_knee)
        )
        return torch.stack((q_abad, q_hip, q_knee), dim=-1)

    def _update_kinematic_trajectory(self):
        """Generate hip/knee cosine feedforward references for stair ascent.

        The reference is intentionally simple: it only biases hip-pitch and
        knee-pitch during the swing phase. PPO actions remain residuals around
        this reference and learn the stabilizing corrections.
        """
        wheel_positions_base = self._get_wheel_positions_base()
        swing_progress, in_swing = self._get_swing_progress()
        stepping = (self.locomotion_mode == 1).unsqueeze(1)
        filtered_force = self.filtered_blocking_forces
        high_force = filtered_force > self.cfg.rewards.squeeze_force_threshold_n
        high_step = self.terrain_step_ahead > self.cfg.rewards.stair_mode_threshold
        descent_edge = (
            self.terrain_step_down_ahead
            > self.cfg.kinematics.feedforward_step_low_m
        )
        low_force = filtered_force < self.cfg.kinematics.feedforward_force_low_n
        low_step = self.terrain_step_ahead < self.cfg.kinematics.feedforward_step_low_m
        obstacle_active = (high_force | high_step) & (~descent_edge)
        locally_clear = (low_force & low_step) | descent_edge
        active_swing = stepping & in_swing & obstacle_active
        new_swing = active_swing & (~self.last_swing_mask)

        for wheel_id in range(len(self.feet_indices)):
            new_ids = new_swing[:, wheel_id]
            if torch.any(new_ids):
                self.wheel_swing_start[new_ids, wheel_id] = \
                    wheel_positions_base[new_ids, wheel_id]
                self.swing_step_target_height[new_ids, wheel_id] = torch.clamp(
                    self.terrain_step_ahead[new_ids, wheel_id],
                    min=self.cfg.rewards.step_up_landing_margin,
                    max=self.cfg.terrain.max_stair_step_height,
                )
                self.trajectory_contact_hold[new_ids, wheel_id] = False
                self.swing_landing_rewarded[new_ids, wheel_id] = False

        phase = swing_progress

        lift_delta = self.cfg.kinematics.feedforward_lift_rate * self.dt
        lower_delta = self.cfg.kinematics.feedforward_lower_rate * self.dt
        lift_state = torch.where(
            stepping & in_swing & obstacle_active,
            self.feedforward_lift_state + lift_delta,
            self.feedforward_lift_state,
        )
        lift_state = torch.where(
            (stepping & in_swing & locally_clear) | (~active_swing),
            lift_state - lower_delta,
            lift_state,
        )
        lift_state = torch.where(
            stepping & in_swing,
            lift_state,
            torch.zeros_like(lift_state),
        )
        self.feedforward_lift_state[:] = torch.clamp(lift_state, 0.0, 1.0)

        blocking_excess = torch.clamp(
            filtered_force - self.cfg.rewards.squeeze_force_threshold_n,
            min=0.0,
        )
        target_amp = (
            self.cfg.kinematics.feedforward_base_amp
            + self.cfg.kinematics.feedforward_force_gain * blocking_excess
            + self.cfg.kinematics.feedforward_height_gain
            * self.terrain_step_ahead
        )
        adaptive_max_amp = torch.clamp(
            target_amp,
            self.cfg.kinematics.feedforward_min_amp,
            self.cfg.kinematics.feedforward_max_amp,
        )
        self.feedforward_amplitudes[:] = (
            self.cfg.kinematics.feedforward_min_amp
            + (
                adaptive_max_amp
                - self.cfg.kinematics.feedforward_min_amp
            )
            * self.feedforward_lift_state
        )

        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        contact_hold = (
            active_swing
            & contacts
            & (phase >= self.cfg.kinematics.contact_hold_phase)
            & low_force
        )
        self.trajectory_contact_hold |= contact_hold
        self.feedforward_lift_state[:] = torch.where(
            self.trajectory_contact_hold,
            torch.zeros_like(self.feedforward_lift_state),
            self.feedforward_lift_state,
        )
        feedforward = torch.where(
            active_swing,
            self.feedforward_amplitudes,
            torch.zeros_like(self.feedforward_amplitudes),
        )
        forward_phase = (
            phase - self.cfg.kinematics.feedforward_forward_start_phase
        ) / (
            self.cfg.kinematics.feedforward_forward_end_phase
            - self.cfg.kinematics.feedforward_forward_start_phase
            + 1e-6
        )
        forward_profile = self._quintic_blend(forward_phase)
        forward_feedforward = torch.where(
            active_swing,
            self.cfg.kinematics.feedforward_forward_amp
            * forward_profile
            * self.feedforward_lift_state,
            torch.zeros_like(forward_profile),
        )

        default_leg_ref = self.default_dof_pos[:, [0, 1, 2, 4, 5, 6]]
        self.kinematic_joint_ref[:] = default_leg_ref
        hip_signs = torch.as_tensor(
            self.cfg.kinematics.feedforward_hip_signs,
            device=self.device,
            dtype=self.kinematic_joint_ref.dtype,
        )
        knee_signs = torch.as_tensor(
            self.cfg.kinematics.feedforward_knee_signs,
            device=self.device,
            dtype=self.kinematic_joint_ref.dtype,
        )
        forward_hip_signs = torch.as_tensor(
            self.cfg.kinematics.feedforward_forward_hip_signs,
            device=self.device,
            dtype=self.kinematic_joint_ref.dtype,
        )
        self.kinematic_joint_ref[:, [1, 4]] += (
            self.cfg.kinematics.feedforward_hip_ratio
            * feedforward
            * hip_signs.unsqueeze(0)
            + forward_feedforward * forward_hip_signs.unsqueeze(0)
        )
        self.kinematic_joint_ref[:, [2, 5]] += (
            self.cfg.kinematics.feedforward_knee_ratio
            * feedforward
            * knee_signs.unsqueeze(0)
        )
        self.wheel_trajectory_ref = wheel_positions_base
        leg_ids = [0, 1, 2, 4, 5, 6]
        self.kinematic_joint_ref[:] = torch.maximum(
            torch.minimum(
                self.kinematic_joint_ref,
                self.dof_pos_limits[leg_ids, 1].unsqueeze(0),
            ),
            self.dof_pos_limits[leg_ids, 0].unsqueeze(0),
        )
        self.last_swing_mask[:] = active_swing

    def _update_locomotion_mode(self):
        """Update rolling/up-step/down-roll modes from proprioceptive signals."""
        wheel_forces_body = torch.zeros_like(
            self.contact_forces[:, self.feet_indices, :]
        )
        for i in range(len(self.feet_indices)):
            wheel_forces_body[:, i, :] = quat_rotate_inverse(
                self.base_quat,
                self.contact_forces[:, self.feet_indices[i], :],
            )

        body_weight = (
            self.cfg.rewards.nominal_robot_mass
            * self.cfg.rewards.gravity_magnitude
        )
        blocking_forces = torch.clamp(
            -wheel_forces_body[:, :, 0], min=0.0
        )
        self.wheel_blocking_forces[:] = blocking_forces
        alpha = self.cfg.kinematics.feedforward_force_filter
        self.filtered_blocking_forces[:] = (
            (1.0 - alpha) * self.filtered_blocking_forces
            + alpha * blocking_forces
        )
        max_blocking_force = torch.max(blocking_forces, dim=1).values
        support_ratio = torch.sum(
            torch.clamp(wheel_forces_body[:, :, 2], min=0.0), dim=1
        ) / body_weight
        squeeze_evidence = (
            max_blocking_force
            > self.cfg.rewards.squeeze_force_threshold_n
        )
        terrain_evidence = (
            torch.max(self.terrain_step_ahead, dim=1).values
            > self.cfg.rewards.stair_mode_threshold
        )
        descent_terrain_evidence = (
            torch.max(self.terrain_step_down_ahead, dim=1).values
            > self.cfg.rewards.stair_mode_threshold
        )
        descent_evidence = (
            support_ratio < self.cfg.rewards.descent_support_force_ratio
        ) & (
            self.root_states[:, 9]
            < -self.cfg.rewards.descent_min_fall_speed
        )

        rolling = self.locomotion_mode == 0
        self.squeeze_confirm_steps = torch.where(
            rolling
            & (squeeze_evidence | terrain_evidence)
            & (~descent_terrain_evidence),
            self.squeeze_confirm_steps + 1,
            torch.zeros_like(self.squeeze_confirm_steps),
        )
        self.descent_confirm_steps = torch.where(
            rolling & (~squeeze_evidence) & descent_evidence,
            self.descent_confirm_steps + 1,
            torch.zeros_like(self.descent_confirm_steps),
        )

        enter_up = rolling & (
            self.squeeze_confirm_steps >= self.squeeze_confirm_limit
        )
        enter_down = rolling & (~enter_up) & self.cfg.rewards.enable_descent_mode & (
            self.descent_confirm_steps >= self.descent_confirm_limit
        )
        if torch.any(enter_up):
            self.locomotion_mode[enter_up] = 1
            self.mode_steps[enter_up] = 0
            self.mode_clear_steps[enter_up] = 0
            # Start with the wheel seeing the larger obstacle/blocked force.
            obstacle_score = (
                self.terrain_step_ahead
                + 0.01 * self.filtered_blocking_forces
            )
            left_blocked = obstacle_score[:, 0] >= obstacle_score[:, 1]
            self.gait_indices[enter_up & left_blocked] = 0.5
            self.gait_indices[enter_up & (~left_blocked)] = 0.0
        if torch.any(enter_down):
            self.locomotion_mode[enter_down] = 2
            self.mode_steps[enter_down] = 0
            self.mode_clear_steps[enter_down] = 0

        active = self.locomotion_mode != 0
        self.mode_steps[active] += 1

        stepping = self.locomotion_mode == 1
        up_clear = (
            max_blocking_force < self.cfg.kinematics.feedforward_force_low_n
        ) & (
            torch.max(self.terrain_step_ahead, dim=1).values
            < self.cfg.kinematics.feedforward_step_low_m
        )
        self.mode_clear_steps = torch.where(
            stepping & up_clear,
            self.mode_clear_steps + 1,
            torch.where(
                stepping,
                torch.zeros_like(self.mode_clear_steps),
                self.mode_clear_steps,
            ),
        )
        exit_up = (
            stepping
            & (self.mode_steps >= self.stair_mode_min_steps)
            & (self.mode_clear_steps >= self.stair_mode_exit_clear_limit)
        )

        descending = self.locomotion_mode == 2
        descent_stable = (
            support_ratio > 0.6
        ) & (torch.abs(self.root_states[:, 9]) < 0.05)
        self.mode_clear_steps = torch.where(
            descending & descent_stable,
            self.mode_clear_steps + 1,
            torch.where(
                descending,
                torch.zeros_like(self.mode_clear_steps),
                self.mode_clear_steps,
            ),
        )
        exit_down = (
            descending
            & (self.mode_steps >= self.descent_mode_min_steps)
            & (self.mode_clear_steps >= self.descent_mode_exit_stable_limit)
        )
        exit_mode = exit_up | exit_down
        self.locomotion_mode[exit_mode] = 0
        self.mode_steps[exit_mode] = 0
        self.mode_clear_steps[exit_mode] = 0
    
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        self._step_contact_targets()
        self._update_terrain_step_ahead()
        self._update_locomotion_mode()
        self._update_kinematic_trajectory()

        descending = self.locomotion_mode == 2
        self.commands[descending, 0] = torch.clamp(
            self.commands[descending, 0],
            max=self.cfg.rewards.descent_max_command_speed,
        )

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
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

        zero_mask = torch.rand(len(env_ids), device=self.device) < \
            self.cfg.commands.zero_command_prob
        zero_env_ids = env_ids[zero_mask]
        self.commands[zero_env_ids, :3] = 0.0
        if self.cfg.commands.heading_command and len(zero_env_ids) > 0:
            forward = quat_apply(
                self.base_quat[zero_env_ids], self.forward_vec[zero_env_ids]
            )
            self.commands[zero_env_ids, 3] = torch.atan2(
                forward[:, 1], forward[:, 0]
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
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:] = 0.0  # previous actions
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)
        # 0: continuous rolling, 1: squeeze-triggered stair stepping,
        # 2: compliant low-speed descent.
        self.locomotion_mode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.mode_steps = torch.zeros_like(self.locomotion_mode)
        self.squeeze_confirm_steps = torch.zeros_like(self.locomotion_mode)
        self.descent_confirm_steps = torch.zeros_like(self.locomotion_mode)
        self.mode_clear_steps = torch.zeros_like(self.locomotion_mode)
        self.squeeze_confirm_limit = max(
            1, int(self.cfg.rewards.squeeze_confirm_frames)
        )
        self.descent_confirm_limit = max(
            1, int(np.ceil(self.cfg.rewards.descent_confirm_time_s / self.dt))
        )
        self.stair_mode_min_steps = max(
            1, int(np.ceil(self.cfg.rewards.stair_mode_min_time_s / self.dt))
        )
        self.stair_mode_exit_clear_limit = max(
            1,
            int(
                np.ceil(
                    self.cfg.rewards.stair_mode_exit_clear_time_s / self.dt
                )
            ),
        )
        self.descent_mode_min_steps = max(
            1,
            int(np.ceil(self.cfg.rewards.descent_mode_min_time_s / self.dt)),
        )
        self.descent_mode_exit_stable_limit = max(
            1,
            int(
                np.ceil(
                    self.cfg.rewards.descent_mode_exit_stable_time_s / self.dt
                )
            ),
        )
        self.kinematic_joint_ref = self.default_dof_pos[:, [0, 1, 2, 4, 5, 6]].clone()
        initial_wheel_positions = self._get_wheel_positions_base()
        self.wheel_swing_start = initial_wheel_positions.clone()
        self.wheel_swing_target = initial_wheel_positions.clone()
        self.wheel_trajectory_ref = initial_wheel_positions.clone()
        self.last_swing_mask = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.trajectory_contact_hold = torch.zeros_like(self.last_swing_mask)
        self.wheel_blocking_forces = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.feedforward_amplitudes = torch.zeros_like(self.wheel_blocking_forces)
        self.feedforward_lift_state = torch.zeros_like(self.wheel_blocking_forces)
        self.filtered_blocking_forces = torch.zeros_like(self.wheel_blocking_forces)
        self.terrain_step_ahead = torch.zeros_like(self.wheel_blocking_forces)
        self.terrain_step_down_ahead = torch.zeros_like(self.wheel_blocking_forces)
        self.swing_step_target_height = torch.zeros_like(self.wheel_blocking_forces)
        self.current_feet_air_time = torch.zeros_like(self.wheel_blocking_forces)
        self.swing_landing_rewarded = torch.zeros_like(self.last_swing_mask)

    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

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

    def _reward_action_transition_smooth(self):
        """Apply extra action smoothing around lift-off and touchdown."""
        action_curvature = torch.square(
            self.actions
            - 2 * self.last_actions[:, :, 0]
            + self.last_actions[:, :, 1]
        )
        stair_gate = self._get_stair_mode_gate()
        transition_gate = stair_gate * 4.0 * self.desired_contact_states * (
            1.0 - self.desired_contact_states
        )
        actions_per_leg = self.num_actions // len(self.feet_indices)
        transition_gate = transition_gate.repeat_interleave(actions_per_leg, dim=1)
        return torch.sum(transition_gate * action_curvature, dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_stair_progress(self):
        """Reward forward progress along the straight stair direction."""
        forward_speed = (
            self.base_position[:, 0] - self.last_base_position[:, 0]
        ) / self.dt
        return torch.clip(
            forward_speed, 0.0, self.cfg.commands.ranges.lin_vel_x[1]
        )

    def _reward_stair_progress_stable(self):
        """Reward forward stair progress only when body attitude is usable."""
        forward_speed = (
            self.base_position[:, 0] - self.last_base_position[:, 0]
        ) / self.dt
        progress = torch.clip(
            forward_speed, 0.0, self.cfg.commands.ranges.lin_vel_x[1]
        )
        attitude_gate = torch.exp(
            -4.0 * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        )
        collision_gate = (
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
            < 1.0
        ).all(dim=1).float()
        return self._get_stair_mode_gate().squeeze(1) * progress * attitude_gate * collision_gate

    def _reward_lateral_motion(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_yaw_motion(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_lateral_position(self):
        lateral_error, _ = self._get_path_errors()
        return torch.square(lateral_error)

    def _reward_heading_error(self):
        _, heading_error = self._get_path_errors()
        effective_error = torch.clamp(
            torch.abs(heading_error) - self.cfg.rewards.heading_error_deadband,
            min=0.0,
        )
        return torch.square(effective_error)

    def _reward_swing_wheel_clearance(self):
        swing_progress, in_swing = self._get_swing_progress()
        peak_height = self._get_adaptive_clearance_peaks()
        stair_gate = self._get_stair_mode_gate(peak_height)
        swing_mask = (
            stair_gate
            * (1.0 - self.desired_contact_states)
            * in_swing.float()
        )
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

    def _reward_swing_wheel_forward(self):
        swing_progress, in_swing = self._get_swing_progress()
        peak_height = self._get_adaptive_clearance_peaks()
        stair_gate = self._get_stair_mode_gate(peak_height)
        swing_mask = (
            stair_gate
            * (1.0 - self.desired_contact_states)
            * in_swing.float()
        )
        phase_profile = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        phase_profile = torch.pow(
            phase_profile, self.cfg.rewards.feet_height_phase_power
        )
        target_height = peak_height * phase_profile
        height_gate = torch.exp(
            -torch.square(self.foot_heights - target_height)
            / self.cfg.rewards.feet_clearance_sigma
        )
        forward_vel = torch.clip(
            self.foot_relative_velocities[:, :, 0],
            0.0,
            self.cfg.rewards.swing_forward_vel_target,
        )
        return torch.sum(
            swing_mask * height_gate * forward_vel, dim=1
        ) / (torch.sum(swing_mask, dim=1) + 1e-6)

    def _reward_swing_foot_forward(self):
        """Point-foot-style forward swing reward for wheel-foot stair learning."""
        return self._reward_swing_wheel_forward()

    def _reward_swing_step_forward_position(self):
        """Reward swing wheel moving forward far enough relative to lift-off."""
        swing_progress, in_swing = self._get_swing_progress()
        wheel_positions_base = self._get_wheel_positions_base()
        stair_gate = self._get_stair_obstacle_gate(per_leg=True)
        swing_mask = (
            stair_gate
            * (1.0 - self.desired_contact_states)
            * in_swing.float()
        )
        progress_gate = (swing_progress > 0.35).float()
        x_progress = wheel_positions_base[:, :, 0] - self.wheel_swing_start[:, :, 0]
        target = self.cfg.rewards.swing_forward_target
        tracking = torch.exp(
            -torch.square(x_progress - target)
            / self.cfg.rewards.swing_forward_sigma
        )
        positive_gate = (x_progress > 0.02).float()
        return torch.sum(
            swing_mask * progress_gate * positive_gate * tracking, dim=1
        ) / (torch.sum(swing_mask * progress_gate, dim=1) + 1e-6)

    def _reward_swing_step_up_landing(self):
        """Reward first swing touchdown when the wheel lands above lift-off height."""
        _, in_swing = self._get_swing_progress()
        wheel_positions_base = self._get_wheel_positions_base()
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        touchdown = contacts & (~self.last_contacts)
        valid_touchdown = (
            (self.locomotion_mode == 1).unsqueeze(1)
            & (
                self.swing_step_target_height
                >= self.cfg.rewards.step_up_landing_margin
            )
            & touchdown
            & (~self.swing_landing_rewarded)
        )
        z_gain = wheel_positions_base[:, :, 2] - self.wheel_swing_start[:, :, 2]
        target_gain = self.swing_step_target_height
        landing = torch.exp(
            -torch.square(z_gain - target_gain)
            / self.cfg.rewards.step_up_landing_sigma
        )
        landing = landing * (z_gain > self.cfg.rewards.step_up_landing_margin).float()
        reward = torch.sum(valid_touchdown.float() * landing, dim=1)
        self.swing_landing_rewarded |= valid_touchdown
        return reward

    def _reward_stair_contact_number(self):
        """Reward stance contact and swing no-contact during stair stepping."""
        contacts = (self.contact_forces[:, self.feet_indices, 2] > 0.1).float()
        desired = self.desired_contact_states
        stair_gate = self._get_stair_obstacle_gate(per_leg=True)
        match = desired * contacts + (1.0 - desired) * (1.0 - contacts)
        return torch.sum(stair_gate * match, dim=1) / (
            torch.sum(stair_gate, dim=1) + 1e-6
        )

    def _reward_stair_feet_air_time(self):
        """Reward swing wheels staying airborne long enough in stair mode."""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        first_contact = contacts & (~self.last_contacts)
        stair_gate = self._get_stair_obstacle_gate(per_leg=True)
        _, in_swing = self._get_swing_progress()
        air_time = torch.clip(self.current_feet_air_time, max=0.5)
        reward = torch.sum(
            stair_gate * first_contact.float() * in_swing.float() * air_time,
            dim=1,
        )
        return reward

    def _reward_swing_wheel_speed(self):
        """Discourage high wheel spin while airborne for softer touchdown."""
        _, in_swing = self._get_swing_progress()
        stair_gate = self._get_stair_obstacle_gate(per_leg=True)
        wheel_dof_vel = self.dof_vel[:, [3, 7]]
        return torch.sum(
            stair_gate * in_swing.float() * torch.square(wheel_dof_vel), dim=1
        )

    def _reward_wheel_trajectory_tracking(self):
        """Reward tracking of the kinematic wheel-center swing trajectory."""
        _, in_swing = self._get_swing_progress()
        swing_mask = self._get_stair_obstacle_gate(per_leg=True) * in_swing.float()
        wheel_positions_base = self._get_wheel_positions_base()
        position_error = torch.sum(
            torch.square(wheel_positions_base - self.wheel_trajectory_ref),
            dim=-1,
        )
        tracking = torch.exp(
            -position_error / self.cfg.rewards.wheel_trajectory_sigma
        )
        return torch.sum(swing_mask * tracking, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1e-6
        )

    def _reward_feedforward_joint_tracking(self):
        """Weakly encourage the residual policy to preserve the lift scaffold."""
        _, in_swing = self._get_swing_progress()
        swing_mask = self._get_stair_obstacle_gate(per_leg=True) * in_swing.float()
        joint_error = torch.stack(
            (
                self.dof_pos[:, 1] - self.kinematic_joint_ref[:, 1],
                self.dof_pos[:, 2] - self.kinematic_joint_ref[:, 2],
                self.dof_pos[:, 5] - self.kinematic_joint_ref[:, 4],
                self.dof_pos[:, 6] - self.kinematic_joint_ref[:, 5],
            ),
            dim=1,
        )
        joint_mask = torch.cat((swing_mask[:, 0:1].repeat(1, 2),
                                swing_mask[:, 1:2].repeat(1, 2)), dim=1)
        tracking = torch.exp(-torch.square(joint_error) / 0.04)
        return torch.sum(joint_mask * tracking, dim=1) / (
            torch.sum(joint_mask, dim=1) + 1e-6
        )

    def _reward_residual_action(self):
        """Keep PPO corrections small so the kinematic controller stays primary."""
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_descent_vertical_motion(self):
        """Limit downward body speed while rolling off a descending step."""
        descent_gate = (self.locomotion_mode == 2).float()
        downward_speed = torch.clamp(-self.root_states[:, 9], min=0.0)
        return descent_gate * torch.square(downward_speed)

    def _reward_descent_pitch_rate(self):
        """Reduce forward pitching during compliant stair descent."""
        descent_gate = (self.locomotion_mode == 2).float()
        return descent_gate * torch.square(self.base_ang_vel[:, 1])

    def _reward_tracking_contacts_shaped_force(self):
        wheel_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        stair_gate = self._get_stair_obstacle_gate(per_leg=True)
        unwanted_contact = stair_gate * (1.0 - self.desired_contact_states)
        penalty = 1.0 - torch.exp(
            -torch.square(wheel_forces) / self.cfg.rewards.gait_force_sigma
        )
        return torch.sum(unwanted_contact * penalty, dim=1) / len(
            self.feet_indices
        )

    def _reward_tracking_contacts_shaped_vel(self):
        wheel_velocities = torch.norm(self.foot_velocities, dim=-1)
        penalty = 1.0 - torch.exp(
            -torch.square(wheel_velocities) / self.cfg.rewards.gait_vel_sigma
        )
        return torch.sum(
            self.desired_contact_states * penalty, dim=1
        ) / len(self.feet_indices)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (
            (self.foot_heights < self.cfg.rewards.about_landing_threshold)
            & (~contacts)
            & (z_vels < 0.0)
        )
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels)
        )
        return torch.sum(torch.square(landing_z_vels), dim=1)

    def _reward_foot_contact_force(self):
        vertical_force = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2], min=0.0
        )
        body_weight = (
            self.cfg.rewards.nominal_robot_mass
            * self.cfg.rewards.gravity_magnitude
        )
        force_ratio = vertical_force / body_weight
        overload = torch.clamp(
            force_ratio - self.cfg.rewards.contact_force_soft_limit_ratio,
            min=0.0,
        )
        return torch.sum(torch.square(overload), dim=1)

    def _reward_foot_touchdown_impulse(self):
        vertical_force = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2], min=0.0
        )
        contacts = vertical_force > 0.1
        touchdown = contacts & (~self.last_contacts)
        body_weight = (
            self.cfg.rewards.nominal_robot_mass
            * self.cfg.rewards.gravity_magnitude
        )
        normalized_impulse = (
            vertical_force * self.dt
            / (
                body_weight
                * self.cfg.rewards.touchdown_impulse_window_s
                + 1e-6
            )
        )
        excess_impulse = torch.clamp(
            normalized_impulse
            - self.cfg.rewards.touchdown_impulse_soft_limit_ratio,
            min=0.0,
        )
        return torch.sum(
            touchdown.float() * torch.square(excess_impulse), dim=1
        )
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def compute_reward(self):
        """Compute rewards, then preserve contacts for touchdown detection."""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        self.feet_air_time += self.dt
        self.current_feet_air_time[:] = self.feet_air_time
        super().compute_reward()
        self.feet_air_time *= (~contacts).float()
        self.last_contacts[:] = contacts

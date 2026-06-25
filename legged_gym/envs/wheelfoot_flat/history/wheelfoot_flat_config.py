# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym.envs.base.base_config import BaseConfig


class BipedCfgWF(BaseConfig):
    class env:
        num_envs = 8192
        # Base proprioception + gait/mode + feedforward reference/error +
        # compact terrain/contact hints for the actor.
        num_observations = 60
        num_height_samples = 117
        # Critic additionally receives true base velocity (3) and a privileged
        # local terrain-height scan (117). Actor remains proprioceptive-only.
        num_critic_observations = 3 + num_observations + num_height_samples
        num_actions = 8
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "trimesh"  # none, plane, heightfield or trimesh
        horizontal_scale = 0.052  # [m], 0.26 m stair tread = 5 cells
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        curriculum = True
        static_friction = 0.6
        dynamic_friction = 0.6
        restitution = 0.0
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 2
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # Train the first wheel-foot stair policy on straight ascending stairs only.
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        stair_step_height = None
        stair_step_width = 0.26  # [m]
        stair_step_width_scales = [1.0]
        max_stair_step_height = 0.16  # [m]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = False
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1
        zero_command_prob = 0.0

        class ranges:
            lin_vel_x = [0.2, 0.5]  # forward-only stair ascent
            lin_vel_y = [0, 0]  # min max [m/s]
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            # lin_vel_y = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # straight stair ascent first
            heading = [-3.14159, 3.14159]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0.5, 0.5]  # fixed alternating left/right stepping
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            # Actual peak is supplied separately by the adaptive terrain sampler.
            swing_height = [0.0, 0.0]

    class init_state:
        pos = [0.0, 0.0, 0.8 + 0.1664]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }

    class control:
        action_scale_pos = 0.25
        # Roughly 4 rad/s is required for 0.5 m/s with a 0.127 m wheel.
        action_scale_vel = 5.0
        descent_max_wheel_speed = 1.6  # about 0.20 m/s at 0.127 m radius
        descent_extra_leg_damping = 1.0  # additional joint damping [N*m*s/rad]
        residual_joint_scale = 0.10  # PPO residual around feedforward reference [rad]
        residual_wheel_speed_scale = 1.0  # PPO wheel-speed residual [rad/s]
        wheel_velocity_directions = [1.0, 1.0]
        wheel_track_width = 0.335  # midpoint of configured 0.32-0.35 m wheel spacing
        stair_stance_wheel_speed = 1.0  # slow support-wheel rolling [rad/s]
        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42,
            "hip_L_Joint": 42,
            "knee_L_Joint": 42,
            "abad_R_Joint": 42,
            "hip_R_Joint": 42,
            "knee_R_Joint": 42,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "wheel_L_Joint": 0.8,
            "wheel_R_Joint": 0.8,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

    class kinematics:
        # Values derived from mode/WF_TRON1A/urdf/robot.urdf:
        # base->abad=(0.05556, +/-0.105, -0.2602),
        # abad->hip=(-0.077, +/-0.0205, 0),
        # hip->knee=(-0.150, -/+0.0205, -0.25981),
        # knee->wheel=(0.150, +/-0.0435, -0.25981).
        thigh_length = 0.3008
        shank_length = 0.3031
        hip_offset_x = -0.02144
        hip_offset_y = 0.1255
        hip_offset_z = -0.2602
        knee_direction = -1.0
        step_length = 0.22  # forward wheel-center travel per stair step [m]
        step_height_hint = 0.16  # expected upward stair height per triggered step [m]
        apex_clearance = 0.08  # lift above the higher of swing start/target [m]
        lift_end_phase = 0.35
        descent_start_phase = 0.65
        contact_hold_phase = 0.65
        ik_reach_margin = 0.01
        feedforward_min_amp = 0.05  # [rad]
        feedforward_base_amp = 0.20  # [rad]
        feedforward_max_amp = 0.42  # [rad]
        feedforward_force_gain = 0.003  # [rad/N] above squeeze threshold
        feedforward_force_low_n = 15.0  # below this, start lowering [N]
        feedforward_step_low_m = 0.01  # below this, treat stair patch as locally flat [m]
        feedforward_lift_rate = 3.0  # lift-state increase per second
        feedforward_lower_rate = 2.0  # lift-state decrease per second
        feedforward_force_filter = 0.25  # low-pass coefficient for blocking force
        feedforward_height_gain = 1.0  # [rad/m] extra amplitude from step height
        feedforward_hip_ratio = 1.0
        feedforward_knee_ratio = 2.0
        feedforward_forward_amp = 0.10  # explicit swing-leg forward bias [rad]
        feedforward_forward_start_phase = 0.25
        feedforward_forward_end_phase = 0.85
        # URDF hip/knee axes are mirrored left/right. These signs generate the
        # same physical lifting intent on both legs.
        feedforward_hip_signs = [1.0, -1.0]
        feedforward_knee_signs = [-1.0, 1.0]
        feedforward_forward_hip_signs = [-1.0, 1.0]

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/WF_TRON1A/urdf/robot.urdf"
        name = "pointfoot_flat"
        foot_name = "wheel"
        foot_radius = 0.127
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 0.1]
        randomize_base_mass = True
        added_mass_range = [-0.5, 2]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.5
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]
        max_push_vel_xy = 1.0

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 0.0

            # tracking related rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.0
            tracking_lin_vel_pb = 0.0
            tracking_ang_vel_pb = 0.0
            stair_progress = 2.0
            stair_progress_stable = 3.0
            swing_wheel_clearance = 1.5
            swing_wheel_forward = 0.0
            swing_foot_forward = 3.0
            swing_step_forward_position = 3.0
            swing_step_up_landing = 4.0
            stair_contact_number = 2.0
            stair_feet_air_time = 1.0

            # regulation related rewards
            nominal_foot_position = 0.0
            leg_symmetry = 0.2
            # Equal wheel x/z constraints prevent alternating stair steps.
            same_foot_x_position = 0.0
            same_foot_z_position = 0.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.07
            lateral_motion = -1.0
            yaw_motion = -0.5
            lateral_position = -2.0
            heading_error = -2.0
            torques = -0.00010
            dof_acc = -2.5e-7
            action_rate = -0.0125
            dof_pos_limits = -2.0
            collision = -1.0
            action_smooth = -0.015
            action_transition_smooth = -0.02
            orientation = -10.0
            feet_distance = -20.0
            base_height = -2.0
            foot_landing_vel = -0.10
            foot_contact_force = -0.03
            foot_touchdown_impulse = -0.10
            tracking_contacts_shaped_force = -0.8
            # Wheel-center velocity is nonzero during valid rolling, unlike a
            # point foot, so the point-foot stance-velocity penalty is disabled.
            tracking_contacts_shaped_vel = 0.0
            swing_wheel_speed = -0.01
            wheel_trajectory_tracking = 0.0
            feedforward_joint_tracking = 0.5
            residual_action = -0.001
            descent_vertical_motion = 0.0
            descent_pitch_rate = 0.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        heading_error_deadband = 0.03
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.6 + 0.1664
        feet_height_target = 0.20  # fixed stair-mode wheel-bottom clearance [m]
        # Flat terrain uses continuous rolling with both wheels on the ground.
        # A nonzero clearance is commanded only when a step is detected ahead.
        feet_height_target_flat = 0.0
        feet_height_safety_margin = 0.04
        feet_height_lookahead = [0.08, 0.16, 0.24, 0.32]
        stair_mode_threshold = 0.025  # minimum detected step height [m]
        squeeze_force_threshold_n = 30.0  # opposing horizontal wheel force [N]
        squeeze_confirm_frames = 3  # require three consecutive policy frames
        stair_mode_min_time_s = 0.30
        stair_mode_exit_clear_time_s = 0.20
        descent_support_force_ratio = 0.25
        enable_descent_mode = False
        descent_min_fall_speed = 0.05  # [m/s]
        descent_confirm_time_s = 0.04
        descent_mode_min_time_s = 0.30
        descent_mode_exit_stable_time_s = 0.25
        descent_max_command_speed = 0.20  # [m/s]
        feet_height_phase_power = 1.0
        feet_clearance_sigma = 0.01
        swing_forward_vel_target = 0.5
        swing_forward_target = 0.16  # desired swing-wheel forward travel [m]
        swing_forward_sigma = 0.01
        step_up_landing_margin = 0.04  # landing should be above swing start [m]
        step_up_landing_sigma = 0.02
        wheel_trajectory_sigma = 0.03
        min_feet_distance = 0.32
        max_feet_distance = 0.35
        max_contact_force = 100.0  # forces above this value are penalized
        about_landing_threshold = 0.08
        nominal_robot_mass = 20.0
        gravity_magnitude = 9.81
        contact_force_soft_limit_ratio = 1.5
        touchdown_impulse_window_s = 0.02
        touchdown_impulse_soft_limit_ratio = 1.0
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        feet_height_tracking_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 0
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class BipedCfgPPOWF(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = BipedCfgWF.env.num_observations * BipedCfgWF.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 15000  # stair locomotion needs a longer curriculum

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_WF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "WF_TRON1A"
        run_name = ""
        # load and resume
        resume = False
        load_run = "-1"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = "None"  # updated from load_run and chkpt

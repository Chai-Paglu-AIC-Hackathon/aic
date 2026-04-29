#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Wrench

class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Import large libraries and load model in __init__ to avoid overhead during first inference call.
        import torch
        import cv2
        import draccus
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        self.torch = torch
        self.cv2 = cv2
        self.draccus = draccus
        self.ACTPolicy = ACTPolicy
        self.ACTConfig = ACTConfig
        self.load_file = load_file
        self.snapshot_download = snapshot_download

        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Initializing RunACT policy on device: {self.device}")

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        self.repo_id = "chai-paglu/act_14eps_cheatcode"

        # TODO: MAKE SURE TO TRY THIS WITH FRAME ID SET TO `gripper/tcp` INSTEAD OF `base_link` IF PERFORMANCE IS POOR. 
        # THE POLICY WAS TRAINED WITH OBSERVATIONS IN THE TCP FRAME, SO IT MAY EXPECT ACTIONS TO BE COMMANDED IN THE TCP FRAME AS WELL. 
        # THIS REQUIRES A TF TRANSFORM TO BE PUBLISHED FROM `base_link` TO `gripper/tcp`, BUT IT COULD SIGNIFICANTLY IMPROVE PERFORMANCE 
        # BY MATCHING THE TRAINING CONDITIONS OF THE POLICY.

        self.repo_id = "chai-paglu/act_11eps_teleop_sam"

        # Path to your checkpoint folder.
        # Installed ROS package data is placed in package share.
        this_file_parent = Path(__file__).resolve().parent
        policy_path = this_file_parent / "act_14eps_cheatcode"

        DOWNLOAD_FROM_HF_HUB = True
        if not DOWNLOAD_FROM_HF_HUB and not policy_path.exists():
            raise FileNotFoundError(
                f"Policy path not found: {policy_path}. "
                "Ensure the resource folder is installed into package share."
            )

        self.get_logger().info(f"Loading ACT policy from local path: {policy_path}")
        if DOWNLOAD_FROM_HF_HUB:
            policy_path = Path(
                self.snapshot_download(
                    repo_id=self.repo_id,
                    allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
                )
            )
            self.get_logger().info(f"Policy files downloaded to: {policy_path}")
        self.get_logger().info(f"Policy files in directory: {list(policy_path.iterdir())}")

        # Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]
        self.get_logger().info(f"Policy config loaded: {config_dict}")

        config = self.draccus.decode(self.ACTConfig, config_dict)
        self.get_logger().info(f"Policy config decoded successfully: {config}")

        # Prevent torchvision from attempting to download pretrained backbone weights
        if getattr(config, "pretrained_backbone_weights", None) is not None:
            self.get_logger().warn(
                "Overriding pretrained backbone weights to None to avoid internet access"
            )
            config.pretrained_backbone_weights = None

        # Load Policy Architecture & Weights
        self.policy = self.ACTPolicy(config)
        model_weights_path = policy_path / "model.safetensors"
        self.policy.load_state_dict(self.load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

        # -------------------------------------------------------------------------
        # 2. Normalization Stats Loading
        # -------------------------------------------------------------------------
        stats_path = (
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        stats = self.load_file(stats_path)

        # Helper to extract and shape stats for broadcasting
        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        # Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }
        print(f"Image stats: {self.img_stats}")

        # Robot State Stats (1, 26)
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        print(f"Robot state mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

        # Action Stats (1, 7) - Used for Un-normalization
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

        self.get_logger().info("Normalization statistics loaded successfully.")

    def _img_to_tensor(
        self,
        raw_img,
        device: Any,
        scale: float,
        mean: Any,
        std: Any,
    ) -> Any:
        """Converts ROS Image -> Resized -> Permuted -> Normalized Tensor."""
        # 1. Bytes to Numpy (H, W, C)
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        # 2. Resize
        if scale != 1.0:
            img_np = self.cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=self.cv2.INTER_AREA
            )

        # 3. To Tensor -> Permute (HWC -> CHW) -> Float -> Div(255) -> Batch Dim
        tensor = (
            self.torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # 4. Normalize (Apply Mean/Std)
        # Formula: (x - mean) / std
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, Any]:
        """Convert ROS Observation message into dictionary of normalized tensors."""

        # --- Process Cameras ---
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        # --- Process Robot State ---
        # Construct flat state vector (26 dims) matching training order
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        state_np = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # TCP Linear Vel (3)
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                # TCP Angular Vel (3)
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                # TCP Error (6)
                *obs_msg.controller_state.tcp_error,
                # Joint Positions (7)
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

        # Normalize State
        raw_state_tensor = (
            self.torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        start_time = self.time_now()
        # deadline = start_time + Duration(seconds=float(task.time_limit) * 0.5)
        deadline = start_time + Duration(seconds=30.0)

        # Run inference for the specified time limit of the task.
        # The policy is autoregressive and maintains internal state,
        # so we keep calling select_action in a loop until simulation time runs out.
        while self.time_now() < deadline:
            loop_start = self.time_now()

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # 2. Model Inference
            with self.torch.inference_mode():
                # returns shape [1, 7] (first action of chunk)
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            # Formula: (norm * std) + mean
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean

            # 4. Extract and Command
            # raw_action_tensor is [1, 7], taking [0] gives vector of 7
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Action: {action}")

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            frame_id = "base_link"
            if "teleop" in self.repo_id:
                # This policy was trained with TCP observations, so we command in TCP frame.
                frame_id = "gripper/tcp"
            motion_update = self.set_cartesian_twist_target(twist, frame_id=frame_id)
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            # Maintain control rate (approx 4Hz loop = 0.25s sleep) in sim-time.
            elapsed = self.time_now() - loop_start
            elapsed_seconds = max(0.0, elapsed.nanoseconds * 1e-9)
            self.sleep_for(max(0.0, 0.25 - elapsed_seconds))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True

    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        USE_CHEATCODE_PARAMS = True
        USE_TELEOP_PARAMS = False
        if USE_TELEOP_PARAMS:
            motion_update_msg.target_stiffness = np.diag(
                [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
            ).flatten()
            motion_update_msg.target_damping = np.diag(
                [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
            ).flatten()
            motion_update_msg.feedforward_wrench_at_tip = Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
            )
            motion_update_msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif USE_CHEATCODE_PARAMS:
            motion_update_msg.target_stiffness = np.diag(
                [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
            ).flatten()
            motion_update_msg.target_damping = np.diag(
                [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
            ).flatten()
            motion_update_msg.feedforward_wrench_at_tip = Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
            )
            motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return motion_update_msg

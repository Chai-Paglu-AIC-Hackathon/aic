#!/usr/bin/env python3

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

"""
This script is used for teleoperation of the robot end-effector Cartesian pose
using a PS4 joystick controller.
This script uses pygame to read joystick inputs and can be run within the pixi environment.
"""

import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import numpy as np
import pygame
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
    TargetMode,
)
from aic_control_interfaces.srv import (
    ChangeTargetMode,
)
from geometry_msgs.msg import Wrench, Vector3, Twist

SLOW_LINEAR_VEL = 0.02
SLOW_ANGULAR_VEL = 0.03
FAST_LINEAR_VEL = 0.05
FAST_ANGULAR_VEL = 0.05
DEADZONE_THRESHOLD = 0.2

# Joystick axis mappings (PS4 controller)
AXIS_LEFT_X = 0  # Linear X
AXIS_LEFT_Y = 1  # Linear Y
AXIS_RIGHT_X = 3  # Angular Z
AXIS_RIGHT_Y = 4  # Angular Y
AXIS_L2 = 2  # Positive angular X
AXIS_R2 = 5  # Negative angular X

# Button mappings
BUTTON_SQUARE = 0  # Toggle slow mode
BUTTON_CIRCLE = 1  # Toggle fast mode
BUTTON_TRIANGLE = 2  # Toggle TCP frame
BUTTON_CROSS = 3  # Toggle base frame
BUTTON_OPTIONS = 9  # Quit


def initialize_joystick():
    pygame.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected. Please connect a PS4 controller.")

    joystick = pygame.joystick.Joystick(0)
    return joystick


def apply_deadzone(value: float, threshold: float = DEADZONE_THRESHOLD) -> float:
    return value if abs(value) > threshold else 0.0


def read_joystick_axes(joystick):
    pygame.event.pump()

    raw_linear_x = joystick.get_axis(AXIS_LEFT_X)
    raw_linear_y = joystick.get_axis(AXIS_LEFT_Y)
    raw_linear_z = joystick.get_axis(AXIS_RIGHT_Y)
    raw_angular_x = joystick.get_axis(AXIS_RIGHT_X)  # Not used in original logic

    return {
        'linear_x': raw_linear_x,
        'linear_y': raw_linear_y,
        'linear_z': raw_linear_z,
        'angular_x': raw_angular_x,
    }


def read_joystick_triggers(joystick):
    l2 = joystick.get_axis(AXIS_L2)
    r2 = joystick.get_axis(AXIS_R2)

    angular_x = 0.0
    if l2 > 0.5:
        angular_x = 1.0
    elif r2 > 0.5:
        angular_x = -1.0

    return {'angular_x': angular_x}


def read_joystick_buttons(joystick):
    buttons = {}
    for i in range(joystick.get_numbuttons()):
        buttons[i] = joystick.get_button(i)
    return buttons


class AICCartesianJoystickTeleoperatorNode(Node):
    def __init__(self):
        super().__init__("aic_joystick_teleoperator_node")

        # Declare parameters.
        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value

        self.motion_update_publisher = self.create_publisher(
            MotionUpdate, f"/{self.controller_namespace}/pose_commands", 10
        )

        while self.motion_update_publisher.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber to '{self.controller_namespace}/pose_commands'..."
            )
            time.sleep(1.0)

        self.client = self.create_client(
            ChangeTargetMode, f"/{self.controller_namespace}/change_target_mode"
        )

        # Wait for service
        while not self.client.wait_for_service():
            self.get_logger().info(
                f"Waiting for service '{self.controller_namespace}/change_target_mode'..."
            )
            time.sleep(1.0)

        # Initialize pygame and joystick
        joystick = initialize_joystick()
        self.get_logger().info(f"Pygame version: {pygame.version.ver}")
        self.get_logger().info(f"Initialized joystick: {joystick.get_name()}")

        self.joystick = joystick

        # Poll joystick and send commands at 25Hz
        self.timer = self.create_timer(0.04, self.send_references)

        # Variable parameters for teleoperation
        self.linear_vel = FAST_LINEAR_VEL  # Linear velocity (m/s)
        self.angular_vel = FAST_ANGULAR_VEL  # Angular velocity (rad/s)
        self.frame_id = "gripper/tcp"

        # Track button states to detect presses
        self.button_states = {}

    def generate_velocity_motion_update(self, twist, frame_id):

        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        return msg

    def apply_deadzone(self, value: float) -> float:
        return value if abs(value) > DEADZONE_THRESHOLD else 0.0

    def send_references(self):
        axes = read_joystick_axes(self.joystick)
        triggers = read_joystick_triggers(self.joystick)
        buttons = read_joystick_buttons(self.joystick)

        # Read axes
        raw_linear_x = axes['linear_x']
        raw_linear_y = axes['linear_y']
        raw_linear_z = axes['linear_z']
        angular_x_raw = triggers['angular_x']

        print(f"Raw axes - Linear X: {raw_linear_x:.2f}, Linear Y: {raw_linear_y:.2f}, Linear Z: {raw_linear_z:.2f}")

        linear_x = self.apply_deadzone(raw_linear_x) * self.linear_vel
        linear_y = self.apply_deadzone(raw_linear_y) * self.linear_vel
        linear_z = self.apply_deadzone(raw_linear_z) * self.linear_vel

        angular_x = angular_x_raw * self.angular_vel
        angular_y = 0.0
        angular_z = 0.0  # Not mapped, or could use D-pad

        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.linear.z = linear_z
        twist.angular.x = angular_x
        twist.angular.y = angular_y
        twist.angular.z = angular_z

        self.motion_update_publisher.publish(
            self.generate_velocity_motion_update(twist=twist, frame_id=self.frame_id)
        )

        # Make sure zero is published when there is no meaningful joystick input.
        teleop_active = any(
            abs(val) > 0.0
            for val in [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        )

        if teleop_active:
            self.get_logger().info(
                f"Published twist: Translation [{twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}], Angular [{twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f}]"
            )

        # Handle button presses
        if buttons.get(BUTTON_SQUARE, False) and not self.button_states.get(BUTTON_SQUARE, False):
            self.button_states[BUTTON_SQUARE] = True
            self.linear_vel = SLOW_LINEAR_VEL
            self.angular_vel = SLOW_ANGULAR_VEL
            self.get_logger().info(
                f"Activated slow mode: Linear velocity = {self.linear_vel} m/s, angular velocity = {self.angular_vel} rad/s"
            )
        elif not buttons.get(BUTTON_SQUARE, False):
            self.button_states[BUTTON_SQUARE] = False

        if buttons.get(BUTTON_CIRCLE, False) and not self.button_states.get(BUTTON_CIRCLE, False):
            self.button_states[BUTTON_CIRCLE] = True
            self.linear_vel = FAST_LINEAR_VEL
            self.angular_vel = FAST_ANGULAR_VEL
            self.get_logger().info(
                f"Activated fast mode: Linear velocity = {self.linear_vel} m/s, angular velocity = {self.angular_vel} rad/s"
            )
        elif not buttons.get(BUTTON_CIRCLE, False):
            self.button_states[BUTTON_CIRCLE] = False

        if buttons.get(BUTTON_TRIANGLE, False) and not self.button_states.get(BUTTON_TRIANGLE, False):
            self.button_states[BUTTON_TRIANGLE] = True
            self.frame_id = "gripper/tcp"
            self.get_logger().info(f"Toggled target frame_id to '{self.frame_id}'")
        elif not buttons.get(BUTTON_TRIANGLE, False):
            self.button_states[BUTTON_TRIANGLE] = False

        if buttons.get(BUTTON_CROSS, False) and not self.button_states.get(BUTTON_CROSS, False):
            self.button_states[BUTTON_CROSS] = True
            self.frame_id = "base_link"
            self.get_logger().info(f"Toggled target frame_id to '{self.frame_id}'")
        elif not buttons.get(BUTTON_CROSS, False):
            self.button_states[BUTTON_CROSS] = False

        if buttons.get(BUTTON_OPTIONS, False):
            self.get_logger().info("Options button pressed. Shutting down.")
            rclpy.shutdown()

    def send_change_control_mode_req(self, mode):
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode

        self.get_logger().info(f"Sending request to change control mode to {mode}")

        future = self.client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response.success:
            self.get_logger().info(f"Successfully changed control mode to {mode}")
        else:
            self.get_logger().info(f"Failed to change control mode to {mode}")

        time.sleep(0.5)


def main(args=None):

    print(
        f"""
        PS4 Joystick teleoperation for Cartesian control
        ---------------------------
        Left Stick:
            X-axis : Linear X
            Y-axis : Linear Y

        Right Stick:
            X-axis : Angular Y
            Y-axis : Angular X

        Triggers:
            L2/R2 : Linear Z

        Buttons:
            Square : Activate SLOW mode ({SLOW_LINEAR_VEL} m/s and {SLOW_ANGULAR_VEL} rad/s)
            Circle : Activate FAST mode ({FAST_LINEAR_VEL} m/s and {FAST_ANGULAR_VEL} rad/s)
            Triangle : Use TCP ('gripper/tcp') frame
            Cross : Use global ('base_link') frame
            Options : Quit

        Ensure PS4 controller is connected before running.
        """
    )

    try:
        with rclpy.init(args=args):
            node = AICCartesianJoystickTeleoperatorNode()
            if rclpy.ok():
                node.send_change_control_mode_req(TargetMode.MODE_CARTESIAN)
                rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        pygame.quit()
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
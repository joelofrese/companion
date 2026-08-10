"""Connect the CM5 safety loop to ROS 2 and PX4."""

import argparse
import asyncio
import threading

from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_forwarder import Ros2VelocityForwarder
from onboard.safety import LatestDistanceSensor, LatestVelocity


class Ros2SafetyBridge:
    """Connect ROS messages to the CM5 safety service.

    The ROS objects are passed in so the control package does not need ROS 2.
    """

    def __init__(
        self,
        node,
        heartbeat_message,
        setpoint_message,
        distance_message,
        velocity_message,
        qos_profile,
        bind_host: str = "0.0.0.0",
        command_port: int = 5001,
        distance_topic: str = "/fmu/out/distance_sensor",
        velocity_topic: str = "/fmu/out/vehicle_local_position",
        tick_period_s: float = 0.02,
    ):
        self._distance = LatestDistanceSensor()
        self._velocity = LatestVelocity()
        self._receiver = UdpSafetyReceiver(bind_host=bind_host, port=command_port)
        self._forwarder = Ros2VelocityForwarder(
            heartbeat_publisher=node.create_publisher(
                heartbeat_message,
                "/fmu/in/offboard_control_mode",
                qos_profile,
            ),
            setpoint_publisher=node.create_publisher(
                setpoint_message,
                "/fmu/in/trajectory_setpoint",
                qos_profile,
            ),
            heartbeat_factory=heartbeat_message,
            setpoint_factory=setpoint_message,
            timestamp_us=lambda: node.get_clock().now().nanoseconds // 1000,
            heading_provider=self._velocity.heading,
        )
        node.create_subscription(
            distance_message,
            distance_topic,
            self._distance.update,
            qos_profile,
        )
        node.create_subscription(
            velocity_message,
            velocity_topic,
            self._velocity.update,
            qos_profile,
        )
        self._service = SafetyCommandService(
            self._receiver,
            self._forwarder,
            tick_period_s=tick_period_s,
            obstacle_distance=self._distance.read,
            velocity_provider=self._velocity.read,
        )
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    @property
    def port(self) -> int:
        """Return the UDP port."""

        return self._receiver.port

    @property
    def error(self):
        """Return a service error, if one occurred."""

        return self._error

    def start(self):
        """Start the service."""

        if self._thread is not None:
            return
        self._stop.clear()
        self._error = None
        self._service.start()
        self._thread = threading.Thread(target=self._run, name="cm5-safety", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            asyncio.run(self._service.run(self._stop))
        except Exception as error:
            self._error = error

    def close(self):
        """Stop the service."""

        if self._thread is None:
            self._receiver.close()
            return
        self._stop.set()
        self._thread.join()
        self._thread = None


def main(argv=None):
    """Run the bridge on DEXI-OS."""

    parser = argparse.ArgumentParser(description="Run the CM5 ROS 2 safety bridge")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--command-port", type=int, default=5001)
    parser.add_argument("--distance-topic", default="/fmu/out/distance_sensor")
    parser.add_argument(
        "--velocity-topic",
        default="/fmu/out/vehicle_local_position",
    )
    parser.add_argument("--tick-period", type=float, default=0.02)
    args = parser.parse_args(argv)

    import rclpy
    from px4_msgs.msg import (
        DistanceSensor,
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleLocalPosition,
    )
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node("companion_safety_bridge")
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    bridge = Ros2SafetyBridge(
        node,
        heartbeat_message=OffboardControlMode,
        setpoint_message=TrajectorySetpoint,
        distance_message=DistanceSensor,
        velocity_message=VehicleLocalPosition,
        qos_profile=qos_profile,
        bind_host=args.bind_host,
        command_port=args.command_port,
        distance_topic=args.distance_topic,
        velocity_topic=args.velocity_topic,
        tick_period_s=args.tick_period,
    )
    bridge_error = None
    try:
        bridge.start()
        rclpy.spin(node)
    finally:
        bridge.close()
        bridge_error = bridge.error
        node.destroy_node()
        rclpy.shutdown()
    if bridge_error is not None:
        raise bridge_error


if __name__ == "__main__":
    main()

"""Connect the CM5 safety loop to ROS 2 and PX4."""

import argparse
import asyncio
import math
import threading
import time
from numbers import Real

from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_forwarder import Ros2VelocityForwarder


DISTANCE_TIMEOUT_S = 0.15


def _finite_real(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


class LatestDistanceSensor:
    """Keep the newest distance reading; missing data is unsafe."""

    def __init__(self, clock=time.monotonic, timeout_s: float = DISTANCE_TIMEOUT_S):
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, Real)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("distance timeout must be positive")
        self._distance_m = math.nan
        self._clock = clock
        self._timeout_s = timeout_s
        self._updated_at_s = None
        self._lock = threading.Lock()

    def update(self, message):
        distance_m = getattr(message, "current_distance", math.nan)
        minimum_m = getattr(message, "min_distance", None)
        maximum_m = getattr(message, "max_distance", None)
        valid = (
            _finite_real(distance_m)
            and distance_m >= 0.0
            and (minimum_m is None or (_finite_real(minimum_m) and minimum_m >= 0.0))
            and (maximum_m is None or (_finite_real(maximum_m) and maximum_m >= 0.0))
            and (minimum_m is None or maximum_m is None or minimum_m <= maximum_m)
            and (minimum_m is None or distance_m >= minimum_m)
            and (maximum_m is None or distance_m <= maximum_m)
        )
        if not valid:
            distance_m = math.nan
        with self._lock:
            self._distance_m = distance_m
            self._updated_at_s = self._clock()

    def read(self):
        now = self._clock()
        with self._lock:
            if (
                self._updated_at_s is None
                or not _finite_real(now)
                or now < self._updated_at_s
                or now - self._updated_at_s > self._timeout_s
            ):
                return math.nan
            return self._distance_m


class Ros2SafetyBridge:
    """Connect ROS messages to the CM5 safety service.

    The ROS objects are passed in so this Mac package does not need ROS 2.
    """

    def __init__(
        self,
        node,
        heartbeat_message,
        setpoint_message,
        distance_message,
        qos_profile,
        bind_host: str = "0.0.0.0",
        command_port: int = 5001,
        distance_topic: str = "/fmu/out/distance_sensor",
        tick_period_s: float = 0.02,
    ):
        self._distance = LatestDistanceSensor()
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
        )
        node.create_subscription(
            distance_message,
            distance_topic,
            self._distance.update,
            qos_profile,
        )
        self._service = SafetyCommandService(
            self._receiver,
            self._forwarder,
            tick_period_s=tick_period_s,
            obstacle_distance=self._distance.read,
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
    parser.add_argument("--tick-period", type=float, default=0.02)
    args = parser.parse_args(argv)

    import rclpy
    from px4_msgs.msg import DistanceSensor, OffboardControlMode, TrajectorySetpoint
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
        qos_profile=qos_profile,
        bind_host=args.bind_host,
        command_port=args.command_port,
        distance_topic=args.distance_topic,
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

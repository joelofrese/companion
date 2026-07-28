"""ROS 2 composition boundary for the CM5 safety command process."""

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


class LatestDistanceSensor:
    """Thread-safe latest PX4 distance reading; no reading is unsafe."""

    def __init__(self, clock=time.monotonic, timeout_s: float = DISTANCE_TIMEOUT_S):
        if timeout_s <= 0.0:
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
        if (
            isinstance(distance_m, bool)
            or not isinstance(distance_m, Real)
            or not math.isfinite(distance_m)
            or (
                minimum_m is not None
                and (isinstance(minimum_m, bool) or not isinstance(minimum_m, Real)
                     or not math.isfinite(minimum_m))
            )
            or (
                maximum_m is not None
                and (isinstance(maximum_m, bool) or not isinstance(maximum_m, Real)
                     or not math.isfinite(maximum_m))
            )
            or (minimum_m is not None and distance_m < minimum_m)
            or (maximum_m is not None and distance_m > maximum_m)
        ):
            distance_m = math.nan
        with self._lock:
            self._distance_m = distance_m
            self._updated_at_s = self._clock()

    def read(self):
        with self._lock:
            if self._updated_at_s is None or self._clock() - self._updated_at_s > self._timeout_s:
                return math.nan
            return self._distance_m


class Ros2SafetyBridge:
    """Compose ROS publishers and sensor input around the CM5 safety service.

    ROS 2 itself is intentionally injected through a node-like object and
    message classes. This keeps the Mac package importable while giving the
    DEXI process one concrete lifecycle boundary to start and stop.
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
        """Return the bound UDP port after ``start``."""

        return self._receiver.port

    @property
    def error(self):
        """Return a service-thread failure after it has stopped, if any."""

        return self._error

    def start(self):
        """Bind before starting the service thread so senders cannot race it."""

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
        """Stop the service and wait for its final zero command."""

        if self._thread is None:
            self._receiver.close()
            return
        self._stop.set()
        self._thread.join()
        self._thread = None


def main(argv=None):
    """Run the bridge on DEXI-OS where ROS 2 and px4_msgs are installed."""

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
    bridge.start()
    bridge_error = None
    try:
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

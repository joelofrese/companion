"""ROS 2 composition boundary for the CM5 safety command process."""

import asyncio
import math
import threading

from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_forwarder import Ros2VelocityForwarder


class LatestDistanceSensor:
    """Thread-safe latest PX4 distance reading; no reading is unsafe."""

    def __init__(self):
        self._distance_m = math.nan
        self._lock = threading.Lock()

    def update(self, message):
        distance_m = getattr(message, "current_distance", math.nan)
        with self._lock:
            self._distance_m = distance_m

    def read(self):
        with self._lock:
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

    @property
    def port(self) -> int:
        """Return the bound UDP port after ``start``."""

        return self._receiver.port

    def start(self):
        """Bind before starting the service thread so senders cannot race it."""

        if self._thread is not None:
            return
        self._service.start()
        self._thread = threading.Thread(target=self._run, name="cm5-safety", daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.run(self._service.run(self._stop))

    def close(self):
        """Stop the service and wait for its final zero command."""

        if self._thread is None:
            self._receiver.close()
            return
        self._stop.set()
        self._thread.join()
        self._thread = None


def main():
    """Run the bridge on DEXI-OS where ROS 2 and px4_msgs are installed."""

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
    )
    bridge.start()
    try:
        rclpy.spin(node)
    finally:
        bridge.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

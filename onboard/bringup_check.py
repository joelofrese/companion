"""Check the CM5 prerequisites before starting the companion services."""

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def check_module(name: str) -> Check:
    available = importlib.util.find_spec(name) is not None
    return Check(
        f"Python module {name}",
        available,
        "available" if available else "missing",
    )


def check_command(name: str) -> Check:
    path = shutil.which(name)
    return Check(
        f"Command {name}",
        path is not None,
        path or "not found on PATH",
    )


def check_gstreamer_plugin(plugin: str, inspector: str = "gst-inspect-1.0") -> Check:
    if shutil.which(inspector) is None:
        return Check(f"GStreamer plugin {plugin}", False, f"{inspector} not found")
    try:
        result = subprocess.run(
            [inspector, plugin],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(f"GStreamer plugin {plugin}", False, str(error))
    passed = result.returncode == 0
    return Check(
        f"GStreamer plugin {plugin}",
        passed,
        "available" if passed else "not available",
    )


def check_serial_device(path: str) -> Check:
    usable = os.path.exists(path) and os.access(path, os.R_OK | os.W_OK)
    return Check(
        f"Serial device {path}",
        usable,
        "read/write access" if usable else "missing or not readable/writable",
    )


def check_udp_port(port: int, host: str = "0.0.0.0") -> Check:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as error:
        return Check(f"UDP {host}:{port}", False, str(error))
    finally:
        sock.close()
    return Check(f"UDP {host}:{port}", True, "available")


def collect_checks(
    serial_device: str = "/dev/ttyAMA2",
    command_port: int = 5001,
    video_port: int = 5000,
) -> Sequence[Check]:
    return (
        check_module("rclpy"),
        check_module("px4_msgs"),
        check_command("gst-launch-1.0"),
        check_command("gst-inspect-1.0"),
        check_gstreamer_plugin("libcamerasrc"),
        check_gstreamer_plugin("x264enc"),
        check_gstreamer_plugin("rtph264pay"),
        check_serial_device(serial_device),
        check_udp_port(command_port),
        check_udp_port(video_port),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check DEXI-OS companion prerequisites")
    parser.add_argument("--serial-device", default="/dev/ttyAMA2")
    parser.add_argument("--command-port", type=int, default=5001)
    parser.add_argument("--video-port", type=int, default=5000)
    args = parser.parse_args(argv)

    checks = collect_checks(args.serial_device, args.command_port, args.video_port)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

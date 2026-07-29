"""Launch PX4/Gazebo, run the synthetic world, and clean up as one command."""

import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time


BOOT_MARKER = "pxh>"
BOOT_MARKER_BYTES = BOOT_MARKER.encode()
BOOT_TIMEOUT_S = 120.0


def _read_output(process, ready, finished):
    try:
        tail = b""
        while chunk := process.stdout.read(4096):
            if not ready.is_set() and BOOT_MARKER_BYTES in tail + chunk:
                ready.set()
            if not ready.is_set():
                print(f"[PX4] {chunk.decode(errors='replace')}", end="", flush=True)
            tail = (tail + chunk)[-(len(BOOT_MARKER_BYTES) - 1):]
    finally:
        finished.set()


def _stop_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(px4_dir: Path, companion_dir: Path) -> int:
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise RuntimeError("stdbuf is required to observe PX4 boot output")
    if not px4_dir.is_dir():
        raise RuntimeError(f"PX4 directory does not exist: {px4_dir}")

    process = subprocess.Popen(
        [stdbuf, "-oL", "-eL", "make", "px4_sitl", "gz_x500"],
        cwd=px4_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    ready = threading.Event()
    finished = threading.Event()
    output_thread = threading.Thread(
        target=_read_output,
        args=(process, ready, finished),
        daemon=True,
    )
    output_thread.start()
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while not ready.is_set():
            if finished.wait(0.2):
                raise RuntimeError(
                    f"PX4 exited before pxh> (code {process.poll()})"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("PX4 did not reach pxh> before the boot timeout")
        result = subprocess.run(
            [sys.executable, "-u", "-m", "sim.world"],
            cwd=companion_dir,
            check=False,
        )
        return result.returncode
    finally:
        _stop_process_group(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the complete PX4/Gazebo synthetic world")
    parser.add_argument(
        "--px4-dir",
        type=Path,
        default=Path.home() / "Code/Croppie/PX4-Autopilot",
    )
    args = parser.parse_args(argv)
    try:
        return run(args.px4_dir.expanduser().resolve(), Path(__file__).resolve().parent.parent)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"simulation runner: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the Mac-to-CM5 command packet and safety path over local UDP."""

import socket
import time

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver


def run():
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.start()
        sender.sendto(
            CommandPacket(1, VelocityCommand(north_m_s=0.3)).encode(),
            ("127.0.0.1", receiver.port),
        )
        command = VelocityCommand()
        for _ in range(100):
            command = receiver.poll()
            if command == VelocityCommand(north_m_s=0.3):
                break
            time.sleep(0.001)
        if command != VelocityCommand(north_m_s=0.3):
            raise RuntimeError(f"command loopback did not pass the fresh command: {command}")
        obstacle_command = receiver.poll(obstacle_distance_m=0.5)
        if obstacle_command != VelocityCommand(north_m_s=-0.2):
            raise RuntimeError(f"command loopback did not apply obstacle safety: {obstacle_command}")
        print(f"Fresh command={command}; obstacle command={obstacle_command}")
    finally:
        sender.close()
        receiver.close()


if __name__ == "__main__":
    run()

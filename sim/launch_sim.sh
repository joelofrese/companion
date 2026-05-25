#!/bin/bash
# Launch PX4 SITL + Gazebo. Open QGroundControl first.
cd ~/Code/Croppie/PX4-Autopilot
source .venv/bin/activate
make px4_sitl gz_x500

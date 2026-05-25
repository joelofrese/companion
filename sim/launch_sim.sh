#!/bin/bash
# Launch PX4 SITL + Gazebo.
cd ~/Code/Croppie/PX4-Autopilot
source .venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
make px4_sitl gz_x500

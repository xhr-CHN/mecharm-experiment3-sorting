#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
if [ -f /opt/mecharm_ws/install/setup.bash ]; then
    source /opt/mecharm_ws/install/setup.bash
fi

exec "$@"

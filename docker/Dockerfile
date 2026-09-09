FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    find /etc/apt/sources.list.d -type f -name '*.list' -exec \
        sed -i 's|http://packages.ros.org|https://packages.ros.org|g' {} +; \
    for attempt in 1 2 3; do \
        apt-get -o Acquire::Retries=5 update && \
        apt-get -o Acquire::Retries=5 --fix-missing install -y --no-install-recommends \
            fastdds-tools \
            python3-colcon-common-extensions \
            python3-pip \
            python3-opencv \
            ros-humble-cv-bridge \
            ros-humble-control-msgs \
            ros-humble-moveit \
            ros-humble-moveit-configs-utils \
            ros-humble-vision-msgs \
            ros-humble-trajectory-msgs \
            ros-humble-xacro && break; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 15; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade \
        "setuptools>=75.0" \
        "packaging>=24.0" \
    && python3 -m pip install --no-cache-dir ultralytics==8.4.135

# Use the ROS Humble system NumPy/OpenCV copies for cv_bridge compatibility.
# This layer is network-free and avoids relying on PyPI for a NumPy downgrade.
RUN python3 -m pip uninstall -y numpy opencv-python opencv-python-headless || true
ENV PYTHONPATH=/usr/lib/python3/dist-packages

COPY docker/entrypoint.sh /ros_entrypoint_exp3.sh
RUN sed -i 's/\r$//' /ros_entrypoint_exp3.sh \
    && chmod +x /ros_entrypoint_exp3.sh

WORKDIR /workspace/mecharm_exp3
ENTRYPOINT ["/ros_entrypoint_exp3.sh"]
CMD ["bash"]

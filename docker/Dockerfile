FROM tiryoh/ros-desktop-vnc:melodic
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO melodic
ENV USER ubuntu

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python-rosdep python-rosinstall python-catkin-tools python-wstools libsdl-image1.2-dev ros-melodic-rosserial-python ros-melodic-ecl-eigen&& \
    rm -rf /var/lib/apt/lists/*i
RUN curl -kL https://bootstrap.pypa.io/pip/2.7/get-pip.py | python
ENV PATH $PATH:/home/${DOCKER_USER}/.local/bin
RUN pip install chainer==6.0 \
            scikit-image
RUN mkdir -p /home/ubuntu/catkin_ws/src &&\
    cd /home/ubuntu/catkin_ws/src && \
    /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_init_workspace" && \
    cd /home/ubuntu/catkin_ws && \
    /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make"
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "source /home/ubuntu/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ubuntu/catkin_ws" >> ~/.bashrc
RUN echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN cd /home/ubuntu/catkin_ws/src && \
    wget https://raw.githubusercontent.com/open-rdc/nav_cloning/master/nav_cloning.install && \
    wstool init && \
    wstool merge nav_cloning.install && \
    wstool up 
RUN source /opt/ros/melodic/setup.bash && \
    cd /home/ubuntu/catkin_ws && \
    rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y && \
    export CMAKE_PREFIX_PATH=~/catkin_ws/devel:/opt/ros/$ROS_DISTRO
RUN cd /home/ubuntu/catkin_ws && \ 
    /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make"

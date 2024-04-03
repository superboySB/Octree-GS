#
# Docker builder stage.
#
FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04 as builder

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ENV DEBIAN_FRONTEND=noninteractive 
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install

#
# Docker runtime stage.
#
FROM nvcr.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04 as runtime
ENV DEBIAN_FRONTEND=noninteractive 
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5

# Copy all files from /colmap_installed/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
COPY --from=builder /colmap_installed/ /usr/local/

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux vim build-essential wget gedit cmake libopencv-dev curl net-tools
    
# Python
COPY docker/environment.yml /root/environment.yml
RUN mkdir ~/miniconda3 && cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm miniconda3/miniconda.sh && \
    miniconda3/bin/conda init bash
RUN /root/miniconda3/bin/conda env create -f /root/environment.yml

WORKDIR /workspace
CMD ["/bin/bash && conda activate gsfast"]

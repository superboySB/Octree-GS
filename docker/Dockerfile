FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive 
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6"

RUN apt update && apt install -y tzdata curl wget vim locate git && apt clean && rm -rf /var/lib/apt/lists/*

# Install nerfstudio
WORKDIR /workspace
RUN git clone https://github.com/nerfstudio-project/nerfstudio.git
RUN cd nerfstudio && pip install --upgrade pip setuptools && pip install -e .

# Build colmap first
WORKDIR /workspace
RUN apt update && apt install -y imagemagick
RUN apt update && apt install -y libgl1-mesa-dev libcgal-dev libboost-all-dev libflann-dev qt5-default libmetis-dev \
    libglew-dev libfreeimage-dev libceres-dev && apt clean && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/colmap/colmap
RUN mkdir colmap/build
WORKDIR /workspace/colmap/build
RUN echo "set_target_properties(colmap_main PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE \
    INSTALL_RPATH /usr/lib/x86_64-linux-gnu:/opt/conda/lib)" >> ../src/colmap/exe/CMakeLists.txt
RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES="60;70;80;86" #-DCUDA_ENABLED=OFF 
RUN make -j8
RUN make install
RUN echo "export PATH=$PATH:/workspace/colmap/build/src/colmap/exe" >> /root/.bashrc

# dependencies of original 3DGS
WORKDIR /workspace
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
WORKDIR /workspace/gaussian-splatting
RUN conda env create -f environment.yml

# Tweak the CMake file for matching the existing OpenCV version. Fix the naming of FindEmbree.cmake
WORKDIR /workspace/gaussian-splatting/SIBR_viewers/cmake/linux
RUN sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake
RUN sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake
RUN sed -i 's/\bembree\b/embree3/g' /workspace/gaussian-splatting/SIBR_viewers/src/core/raycaster/CMakeLists.txt
WORKDIR /workspace/gaussian-splatting/SIBR_viewers 
RUN apt update && \
    apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev \
    libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j24 --target install
RUN echo "export PATH=$PATH:/workspace/gaussian-splatting/SIBR_viewers/install/bin" >> /root/.bashrc
RUN echo "export MESA_GL_VERSION_OVERRIDE=4.5" >> /root/.bashrc
RUN apt-get update && \
    apt-get install -y --no-install-recommends git tmux vim wget gedit curl net-tools libpcl-dev pcl-tools && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Dependencies of our project
COPY docker/environment.yml /root/environment.yml
RUN conda env create -f /root/environment.yml && conda init bash
RUN echo "conda activate gsfast" >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]


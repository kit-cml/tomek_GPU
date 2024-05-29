# Stage 1: Build environment with CUDA and necessary tools (Ubuntu-based)
# we need this to be driver version specific
FROM nvidia/cuda:12.2.2-devel-ubuntu20.04 AS builder

WORKDIR /app

# Set non-interactive environment
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools (Ubuntu-specific packages)
RUN apt-get update && apt-get install -y \
    make \
    cmake \
    nano \
    libtool \
    libjson-c-dev \
    libcurl4-openssl-dev \
    libc6-dev-i386 \
    wget \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Download and install cuDNN manually
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.9.5.29-1+cuda12.2_amd64.deb \
    && dpkg -i libcudnn8_8.9.5.29-1+cuda12.2_amd64.deb \
    && rm libcudnn8_8.9.5.29-1+cuda12.2_amd64.deb


# Copy project files
COPY . .

# Compile each project with nvcc
RUN make

# RUN make -C project3  
# Add more commands for additional projects

##------------------------------------------------------------------------------------

# Stage 2: Minimal runtime environment (Ubuntu-based)
# Use runtime to save some space, but if there any error, try to switch to devel
FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
# FROM nvidia/12.2.2-runtime-ubuntu20.04

# Install essential build tools (Ubuntu-specific packages)
RUN apt-get update && apt install glibc-source -y

WORKDIR /app

# Copy compiled binaries from the builder stage
COPY --from=builder /app/bin/drug_sim /app/

# Copy files from the host machine
COPY bin/drugs /app/drugs
COPY bin/result /app/result
COPY bin/input_deck.txt /app/input_deck.txt
COPY bin/run_insilico_postpro.sh /app/run_insilico_postpro.sh
# Run the binary file
# CMD ["./drug_sim"]
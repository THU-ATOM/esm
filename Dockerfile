# Use NVIDIA CUDA 11.7 base image with cuDNN 8
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    build-essential \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ARG python=3.9
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /usr/local/bin/conda tos accept --override-channels --channel https://conda.anaconda.org/pytorch \
    && /usr/local/bin/conda tos accept --override-channels --channel https://conda.anaconda.org/conda-forge \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -y --all

# Configure conda and pip mirrors for faster installation
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN cat <<EOT >> ~/.condarc
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud
EOT


ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
RUN conda install -y \
    pytorch==1.12.1 \
    torchvision==0.13.1 \
    torchaudio=0.12.1 \
    cudatoolkit=11.6 \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    && conda clean -y --all

COPY . /workspace
RUN pip install --no-cache-dir pip==24.0
RUN pip install --no-cache-dir -r /workspace/requirements.txt

WORKDIR /workspace

CMD ["bash"]

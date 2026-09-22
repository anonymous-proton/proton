FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        zlib1g-dev \
        libbz2-dev \
        vim \
        wget \
        bzip2 \
        tmux \
        htop \
        procps \
        xxd \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ENV CUDACXX=/usr/local/cuda/bin/nvcc

WORKDIR /tmp
RUN git clone --recursive https://github.com/soedinglab/MMseqs2.git \
    && cd MMseqs2 \
    && git checkout upstream \
    && git submodule update --recursive \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE -DHAVE_AVX2=1 -DENABLE_CUDA=1 -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" .. \
    && cat CMakeCache.txt | grep -i "CUDA" \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf MMseqs2

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

RUN git config --global --add safe.directory /workspace

RUN pip install --no-cache-dir absl-py
RUN pip install --no-cache-dir importlib-metadata
RUN pip install --no-cache-dir biopython

ARG COLABFOLD_REF=upstream
RUN pip install --no-cache-dir --no-deps "git+https://github.com/sokrypton/ColabFold.git@${COLABFOLD_REF}"

WORKDIR /workspace
ENTRYPOINT []


FROM nvidia/cuda:12.5.0-base-ubuntu22.04

RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet git wget gcc g++ make zlib1g-dev zstd

RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet python3.11 python3-pip python3.11-venv python3.11-dev
RUN python3.11 -m venv /alphafold3_venv
ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"
RUN pip3 install --upgrade pip

RUN mkdir /hmmer_build /hmmer ; \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build ; \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz) ; \
    (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) ; \
    (cd /hmmer_build/hmmer-3.4 && make -j8) ; \
    (cd /hmmer_build/hmmer-3.4 && make install) ; \
    (cd /hmmer_build/hmmer-3.4/easel && make install) ; \
    rm -R /hmmer_build

COPY . /app/alphafold
WORKDIR /app/alphafold

RUN pip3 install -r dev-requirements.txt
RUN pip3 install --no-deps .
RUN build_data

ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["python3", "run_alphafold.py"]

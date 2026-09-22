FROM fovus/vina-gpu-2.1:autodock-vina-gpu

WORKDIR /vina/AutoDock-Vina-GPU-2.1

RUN make source \
    && rm -f Kernel1_Opt.bin Kernel2_Opt.bin

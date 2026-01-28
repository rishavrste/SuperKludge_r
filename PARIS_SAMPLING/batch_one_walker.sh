#!/bin/bash
#PBS -P personal-e1583490
#PBS -k oe
#PBS -N paris
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1


# (optional) load Singularity module if your cluster uses environment modules.
# Uncomment if required on your cluster:
# module load singularity

# Run the commands inside the container. --nv exposes GPUs to the container.
singularity exec --nv -e \
  /app1/common/singularity-img/hopper/cuda/cuda_12.1.0-cudnn8-devel-u20.04.sif \
  bash -lc '

    # Activate environment if present (ignore failure so job still fails later if python missing)
    if command -v conda >/dev/null 2>&1; then
      conda activate few_gpu || echo "Warning: conda activate failed"
    fi

    # optional: check GPUs (uncomment to debug)
    #nvidia-smi || true

    # run the program
    python /home/svu/e1583490/scratch/SuperKludge_r/PARIS_SAMPLING/paris_one_walker.py
  '

# =============================================================================
#  AF2chi / localColabFold - OCI image, built from scratch
#
#  Build:   podman build -f Containerfile.inline -t af2chi:1.0 .
#  Weights: mounted at /cache by default (see CONTAINER.md), or baked in with
#           podman build --build-arg BAKE_PARAMS=true -t af2chi:1.0-full .
#
#  Stage 1 builds the conda environment and patches it with the AF2chi code.
#  Stage 2 copies only the finished environment, so no package caches, no
#  package manager and no build tools end up in the shipped image.
#  Both stages use the same base, so the glibc the environment was built
#  against is the glibc it runs against.
# =============================================================================

ARG CUDA_BASE=docker.io/nvidia/cuda:12.2.2-base-ubuntu22.04
ARG MICROMAMBA_VERSION=2.0.5
ARG AF2CHI_HOME=/opt/af2chi

# -----------------------------------------------------------------------------
# Stage 1: build the environment
# -----------------------------------------------------------------------------
FROM ${CUDA_BASE} AS builder

ARG MICROMAMBA_VERSION
ARG AF2CHI_HOME
ARG BAKE_PARAMS=false

ENV DEBIAN_FRONTEND=noninteractive
ENV ENVDIR=${AF2CHI_HOME}/colabfold-conda
ENV DATADIR=${AF2CHI_HOME}/colabfold

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

# micromamba is only a build-time tool: a single static binary, no base env
RUN curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
      | tar -xj -C /usr/local bin/micromamba && \
    micromamba --version

# --- conda-level dependencies -------------------------------------------------
# Same pins as install_colabbatch_linux.sh. git is needed for the pip git+ URL.
RUN micromamba create -y -p "${ENVDIR}" \
      -c conda-forge -c bioconda \
      git \
      python=3.10 \
      openmm=7.7.0 \
      pdbfixer \
      kalign2=2.04 \
      hhsuite=3.3.0 \
      mmseqs2=15.6f452 \
      setuptools=80.9.0 && \
    micromamba clean --all --yes

ENV PATH=${ENVDIR}/bin:${PATH}

# --- python-level dependencies ------------------------------------------------
# Order mirrors the install script: the AF2chi ColabFold fork goes in first,
# the PyPI pins afterwards resolve as already satisfied.
RUN pip install --no-cache-dir --no-warn-conflicts \
      "colabfold[alphafold-without-jax] @ git+https://github.com/matteo-cagiada/ColabFold-sc" && \
    pip install --no-cache-dir "colabfold==1.5.5" "alphafold-colabfold==2.3.7" && \
    pip install --no-cache-dir --upgrade tensorflow && \
    pip install --no-cache-dir silence_tensorflow && \
    pip install --no-cache-dir --upgrade "flax==0.10.0" "orbax-checkpoint==0.6.0" && \
    pip install --no-cache-dir --upgrade "jax[cuda12_pip]==0.4.28" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# --- AF2chi patch -------------------------------------------------------------
# patcher_colabfold_linux.sh expects <dir>/colabfold-conda/lib/python*/site-packages
# and copies src/af2chi-params into <dir>/colabfold, which must already exist.
RUN mkdir -p "${DATADIR}"
COPY . /src
WORKDIR /src
RUN chmod +x patcher_colabfold_linux.sh && \
    ./patcher_colabfold_linux.sh "${AF2CHI_HOME}"

# --- environment fixups -------------------------------------------------------
# Applied after patching, because the patcher replaces some of these files.
# Each sed is a no-op once its pattern is gone, so this stays idempotent.
RUN cd "${ENVDIR}/lib/python3.10/site-packages/colabfold" && \
    sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py && \
    sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${DATADIR}\"#g" download.py && \
    sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py && \
    grep -q "${DATADIR}" download.py

# --- AlphaFold2 weights -------------------------------------------------------
# Default: leave them out, point the params directory at a /cache mount.
# BAKE_PARAMS=true: download ~4 GB into the image for a self-contained .sif.
RUN if [ "${BAKE_PARAMS}" = "true" ]; then \
        python3 -m colabfold.download ; \
    else \
        ln -s /cache/colabfold/params "${DATADIR}/params" ; \
    fi

# --- slim down ----------------------------------------------------------------
RUN find "${ENVDIR}" -follow -type f -name '*.a' -delete && \
    find "${ENVDIR}" -follow -type f -name '*.pyc' -delete && \
    find "${ENVDIR}" -follow -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    rm -rf /src

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM ${CUDA_BASE}

ARG AF2CHI_HOME

LABEL org.opencontainers.image.title="AF2chi (localColabFold)" \
      org.opencontainers.image.description="Side-chain rotamer distributions and structural ensembles from AlphaFold2" \
      org.opencontainers.image.source="https://github.com/matteo-cagiada/AF2chi_localcolabfold" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${AF2CHI_HOME} ${AF2CHI_HOME}

# LD_LIBRARY_PATH covers the GCC/libstdc++ errors described in the README:
# openmm and hhsuite link against the libraries shipped inside the env.
ENV PATH=${AF2CHI_HOME}/colabfold-conda/bin:${PATH} \
    LD_LIBRARY_PATH=${AF2CHI_HOME}/colabfold-conda/lib \
    MPLBACKEND=agg \
    PYTHONUNBUFFERED=TRUE

# Build-time smoke test: the CLI imports and the AF2chi flags are present.
RUN colabfold_batch --help > /tmp/help.txt && \
    grep -q -- "--af2chi-backbone" /tmp/help.txt && \
    grep -q -- "--af2chi-ensemble" /tmp/help.txt && \
    rm /tmp/help.txt

# Entrypoint: prepares the weights directory so ColabFold's own first-run
# download works against whatever is mounted at /cache.
COPY af2chi-entrypoint.sh /usr/local/bin/af2chi-entrypoint.sh
RUN chmod +x /usr/local/bin/af2chi-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/af2chi-entrypoint.sh", "colabfold_batch"]
WORKDIR /work
CMD ["--help"]

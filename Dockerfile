FROM ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

# This is done in the install_colabbatch_linux.sh script, but we will mount a colabfold cache directory from the cluster
#RUN /usr/local/envs/colabfold/bin/python3 -m colabfold.download

# create fake "localcolabfold" directory
# this is needed to make the patcher script work
RUN mkdir -p /app/localcolabfold
RUN ln -s /usr/local/envs/colabfold/ /app/localcolabfold/colabfold-conda

ENV COLABFOLDDIR=/app/localcolabfold

# modify the default params directory, required for AF2chi
RUN cd "${COLABFOLDDIR}/colabfold-conda/lib/python3.9/site-packages/colabfold" && \
    sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py

# patch colabfold and alphafold
COPY src/colabfold-conda-files/ ${COLABFOLDDIR}/colabfold-conda/lib/python3.9/site-packages/

# Add af2chi parameters
ADD src/af2chi-params ${COLABFOLDDIR}/colabfold/af2chi-params

# Add symlink for AlphaFold model parameters, colabfold image uses /cache directory for them.
RUN ln -s /cache/colabfold/params ${COLABFOLDDIR}/colabfold/

ENV MPLBACKEND agg

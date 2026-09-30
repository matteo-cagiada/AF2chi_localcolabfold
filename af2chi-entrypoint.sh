#!/bin/sh
# =============================================================================
#  AF2chi container entrypoint
#
#  The AlphaFold2 weights are not part of the image. There are three ways to
#  provide them, and this script makes all of them work:
#
#   1. Mount a directory at /cache. The weights are downloaded into it on the
#      first run (~4 GB) and reused afterwards:
#          -v $HOME/colabfold-cache:/cache
#
#   2. Mount weights you already have and point AF2chi at them:
#          -v /shared/alphafold:/weights   ... colabfold_batch --data /weights
#
#   3. Bake them into the image at build time (BAKE_PARAMS=true), in which
#      case nothing needs mounting.
#
#  The image symlinks <data dir>/params to /cache/colabfold/params. ColabFold
#  creates that directory with mkdir(exist_ok=True), which fails on a dangling
#  symlink, so the target is created here before ColabFold runs.
# =============================================================================

WEIGHTS_TARGET="${AF2CHI_WEIGHTS:-/cache/colabfold/params}"

# Skip all of this if the caller passed --data: they are pointing somewhere else.
uses_data_flag=0
for arg in "$@"; do
    case "$arg" in
        --data|--data=*) uses_data_flag=1 ;;
    esac
done

if [ "$uses_data_flag" -eq 0 ] && [ ! -e "$WEIGHTS_TARGET" ]; then
    if mkdir -p "$WEIGHTS_TARGET" 2>/dev/null; then
        cat <<EOF
[af2chi] AlphaFold2 weights not found in $WEIGHTS_TARGET
[af2chi] They will be downloaded now (~4 GB). This happens once, as long as
[af2chi] the directory mounted there persists between runs.
EOF
    else
        cat <<EOF
[af2chi] WARNING: cannot write to $WEIGHTS_TARGET
[af2chi]
[af2chi] Nothing writable is mounted at /cache, so the weights would be
[af2chi] downloaded into the container and lost when it exits. Either:
[af2chi]   mount a writable directory:   -v /your/cache:/cache
[af2chi]   or use weights you have:      --data /path/to/weights
EOF
    fi
fi

exec "$@"

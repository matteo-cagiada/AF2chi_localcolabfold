#!/usr/bin/env bash
# =============================================================================
#  run_af2chi.sh - run the AF2chi container with docker, podman or apptainer
#
#  Everything after the script name is passed straight to colabfold_batch, and
#  the current directory is mounted as the working directory, so paths work
#  exactly as they would in a native install:
#
#      ./run_af2chi.sh --af2chi-backbone templates/ input.fasta results/
#      ./run_af2chi.sh --af2chi-ensemble templates/ input.fasta results/ \
#          --n-struct-ensemble 50
#      ./run_af2chi.sh --help
#
#  Options, all optional, which must come BEFORE the colabfold_batch options:
#
#    --cache PATH       where the AlphaFold2 weights live on the host
#    --image REF        image reference, or path to a .sif
#    --engine NAME      docker | podman | apptainer
#    --mount HOST:CONT  extra bind mount, repeatable
#    --cpu              run without a GPU
#    --dry-run          print the command instead of running it
#
#  The same settings as environment variables:
#
#    AF2CHI_CACHE    where the AlphaFold2 weights live on the host.
#                    Downloaded there on the first run (~4 GB) and reused.
#                    [default: $HOME/.cache/colabfold]
#    AF2CHI_IMAGE    image reference, or path to a .sif for apptainer
#                    [default: af2chi:1.0, or ./af2chi.sif for apptainer]
#    AF2CHI_ENGINE   docker | podman | apptainer   [default: first one found]
#    AF2CHI_GPU      0 to run on CPU                [default: 1]
#    AF2CHI_DRYRUN   1 to print the command instead of running it
#
#  Example: weights on a shared filesystem rather than your home directory
#
#      AF2CHI_CACHE=/ceph/group/alphafold-weights ./run_af2chi.sh --af2chi in.fasta out/
#
#  Notes
#  -----
#  * The directory you launch from is mounted as the container's working
#    directory, so relative paths behave as they do in a native install. An
#    argument that points outside it (../data/in.fasta, or an absolute path such
#    as --data /shared/weights) is detected and bound as well, with the argument
#    rewritten to the path it has inside the container.
#  * Engine autodetection prefers podman. On a host with both podman and
#    apptainer, pass --engine apptainer to use a .sif.
# =============================================================================

set -euo pipefail

if [ "$#" -eq 0 ]; then
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

# --- wrapper flags, which must come before the colabfold_batch options -------
CLI_CACHE=""; CLI_IMAGE=""; CLI_ENGINE=""; CLI_GPU=""; CLI_DRYRUN=""
CLI_MOUNTS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cache)   CLI_CACHE="$2";  shift 2 ;;
        --cache=*) CLI_CACHE="${1#*=}"; shift ;;
        --image)   CLI_IMAGE="$2";  shift 2 ;;
        --image=*) CLI_IMAGE="${1#*=}"; shift ;;
        --engine)  CLI_ENGINE="$2"; shift 2 ;;
        --engine=*) CLI_ENGINE="${1#*=}"; shift ;;
        --mount)   CLI_MOUNTS+=("$2"); shift 2 ;;
        --mount=*) CLI_MOUNTS+=("${1#*=}"); shift ;;
        --cpu)     CLI_GPU=0; shift ;;
        --dry-run) CLI_DRYRUN=1; shift ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done

if [ "$#" -eq 0 ]; then
    echo "[ERROR] No colabfold_batch options given." >&2
    echo "        e.g. $0 --cache /scratch/weights --af2chi-backbone templates/ in.fasta out/" >&2
    exit 1
fi

# --- engine ------------------------------------------------------------------
ENGINE="${CLI_ENGINE:-${AF2CHI_ENGINE:-}}"
if [ -z "$ENGINE" ]; then
    for candidate in podman docker apptainer singularity; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ENGINE="$candidate"
            break
        fi
    done
fi
if [ -z "$ENGINE" ]; then
    echo "[ERROR] No container engine found (looked for podman, docker, apptainer)." >&2
    exit 1
fi

# --- image -------------------------------------------------------------------
case "$ENGINE" in
    apptainer|singularity) DEFAULT_IMAGE="./af2chi.sif" ;;
    *)                     DEFAULT_IMAGE="ghcr.io/matteo-cagiada/af2chi:1.0" ;;
esac
IMAGE="${CLI_IMAGE:-${AF2CHI_IMAGE:-$DEFAULT_IMAGE}}"

# --- weights cache -----------------------------------------------------------
CACHE="${CLI_CACHE:-${AF2CHI_CACHE:-$HOME/.cache/colabfold}}"
mkdir -p "$CACHE"
CACHE="$(cd "$CACHE" && pwd)"

if [ ! -d "$CACHE/colabfold/params" ]; then
    echo "[af2chi] Weights directory: $CACHE"
    echo "[af2chi] The AlphaFold2 weights (~4 GB) will be downloaded there on this run."
    echo "[af2chi] Use --cache PATH to put them somewhere else."
fi

WORKDIR="$(pwd)"
GPU="${CLI_GPU:-${AF2CHI_GPU:-1}}"

# --- make paths outside the working directory reachable ----------------------
# Only the working directory is mounted, so an argument such as
# ../data/input.fasta would not exist inside the container. Find those, bind
# the directory holding them at the same absolute path, and rewrite the
# argument to match, so relative paths that climb out still work.

abspath() {
    if realpath -m / >/dev/null 2>&1; then
        realpath -m "$1"
    else
        python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
    fi
}

NEW_ARGS=()
AUTO_DIRS=()
for arg in "$@"; do
    # options and option=value pairs are never paths
    case "$arg" in
        -*) NEW_ARGS+=("$arg"); continue ;;
    esac

    # treat the argument as a path only if it exists, or looks like one whose
    # parent directory exists: this leaves values such as "1,2" or "50" alone
    if [ ! -e "$arg" ]; then
        case "$arg" in
            */*) if [ ! -d "$(dirname "$arg")" ]; then NEW_ARGS+=("$arg"); continue; fi ;;
            *)   NEW_ARGS+=("$arg"); continue ;;
        esac
    fi

    resolved="$(abspath "$arg")"
    case "$resolved/" in
        "$WORKDIR"/*) NEW_ARGS+=("$arg"); continue ;;   # already inside the mount
    esac

    if [ -d "$resolved" ]; then
        host_dir="$resolved"
    else
        host_dir="$(dirname "$resolved")"
    fi
    AUTO_DIRS+=("$host_dir")
    NEW_ARGS+=("$resolved")
done

# drop any directory already covered by another one in the list, so a path and
# its parent do not both get bound
n_auto=0
for d in ${AUTO_DIRS+"${AUTO_DIRS[@]}"}; do
    covered=0
    for e in ${AUTO_DIRS+"${AUTO_DIRS[@]}"}; do
        if [ "$d" = "$e" ]; then continue; fi
        case "$d/" in
            "$e"/*) covered=1 ;;
        esac
    done
    if [ "$covered" -eq 1 ]; then continue; fi
    already=0
    for m in ${CLI_MOUNTS+"${CLI_MOUNTS[@]}"}; do
        if [ "${m%%:*}" = "$d" ]; then already=1; fi
    done
    if [ "$already" -eq 1 ]; then continue; fi
    CLI_MOUNTS+=("$d:$d")
    n_auto=$((n_auto + 1))
    echo "[af2chi] $d is outside the working directory: binding it as well"
done

if [ "$n_auto" -gt 0 ]; then
    echo "[af2chi] (launch from a directory holding every input and output to avoid this)"
fi
set -- ${NEW_ARGS+"${NEW_ARGS[@]}"}

# --- build the command -------------------------------------------------------
case "$ENGINE" in
    podman)
        CMD=(podman run --rm -it
             -v "$CACHE":/cache:z
             -v "$WORKDIR":/work:z
             -w /work)
        for m in ${CLI_MOUNTS+"${CLI_MOUNTS[@]}"}; do CMD+=(-v "$m:z"); done
        [ "$GPU" = "1" ] && CMD+=(--device nvidia.com/gpu=all)
        CMD+=("$IMAGE" colabfold_batch "$@")
        ;;
    docker)
        CMD=(docker run --rm -it
             -u "$(id -u):$(id -g)"
             -v "$CACHE":/cache
             -v "$WORKDIR":/work
             -w /work)
        for m in ${CLI_MOUNTS+"${CLI_MOUNTS[@]}"}; do CMD+=(-v "$m"); done
        [ "$GPU" = "1" ] && CMD+=(--gpus all)
        CMD+=("$IMAGE" colabfold_batch "$@")
        ;;
    apptainer|singularity)
        if [ ! -f "$IMAGE" ]; then
            echo "[ERROR] Image file not found: $IMAGE" >&2
            echo "        Build it with: apptainer build af2chi.sif docker://ghcr.io/<owner>/af2chi:1.0" >&2
            echo "        or set AF2CHI_IMAGE to its path." >&2
            exit 1
        fi
        mkdir -p "$CACHE/colabfold/params"   # apptainer images are read-only
        CMD=("$ENGINE" run
             -B "$CACHE":/cache
             -B "$WORKDIR":/work
             --pwd /work)
        for m in ${CLI_MOUNTS+"${CLI_MOUNTS[@]}"}; do CMD+=(-B "$m"); done
        [ "$GPU" = "1" ] && CMD+=(--nv)
        CMD+=("$IMAGE" "$@")
        ;;
esac

if [ "${CLI_DRYRUN:-${AF2CHI_DRYRUN:-0}}" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

echo "[af2chi] engine: $ENGINE   image: $IMAGE"
exec "${CMD[@]}"

#!/usr/bin/env python3
"""
traj2af2chi.py — build an AF2chi template ensemble from an MD trajectory,
a multi-model PDB, or a folder of PDB files.

The output is a directory of AF2-compatible mmCIF files, named with 4
lowercase characters (x001.cif, x002.cif, ...), ready to hand to:

    colabfold_batch --af2chi-ensemble <output_dir> input.fasta results/

Examples
--------
    # every 10th frame of a trajectory, protein only
    traj2af2chi.py -x traj.xtc -t topol.tpr -s 10 -o templates/

    # a frame range, superimposed on the first frame
    traj2af2chi.py -x traj.xtc -t topol.gro --start 1000 --stop 2000 -s 5 \
        --align -o templates/

    # a multi-model PDB (NMR models, or a pre-extracted ensemble)
    traj2af2chi.py -p models.pdb -o templates/

    # a folder of single-structure PDB files
    traj2af2chi.py -p pdbs/ -o templates/

Dependencies
------------
    Biopython           always required (already present in the AF2chi env)
    MDAnalysis          only required for trajectory input (-x/-t)

MDAnalysis is imported lazily, so PDB inputs work inside the AF2chi
environment without installing anything extra. For trajectory input, use a
separate lightweight environment (see tools/environment.yml) rather than
adding MDAnalysis to the AF2chi environment.
"""

import argparse
import string
import sys
import tempfile
import textwrap
from collections import defaultdict
from datetime import date
from pathlib import Path

try:
    from Bio.PDB import PDBParser, MMCIFIO
except ImportError:
    sys.exit(
        "[ERROR] Biopython is not installed.\n"
        "        Install it with:  pip install biopython"
    )

# Standard three-letter to one-letter map. Inlined so this script does not
# depend on the alphafold package (it used only this dictionary).
RESTYPE_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Protonation-state and force-field variants that MD engines emit, plus
# selenomethionine. AF2 knows none of these, so they are folded back onto the
# standard residue in the coordinates themselves, not just in the sequence:
# AF2's template parser reads _entity_poly_seq and _atom_site, and an unknown
# residue name there costs you the template.
RESNAME_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "ARN": "ARG", "TYM": "TYR", "MSE": "MET",
}

# Capping groups: not residues, dropped rather than renamed.
DROP_RESNAMES = {"ACE", "NME", "NMA", "NH2"}


def normalize_pdb_text(text: str):
    """Rewrite MD residue names to the standard set AF2 understands.

    Aliased residues are renamed in place and promoted from HETATM to ATOM;
    selenomethionine's SE atom becomes SD; capping groups are dropped.
    Returns (new_text, counts).
    """
    out, counts = [], {}
    for line in text.splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            out.append(line)
            continue

        resname = line[17:20].strip().upper()

        if resname in DROP_RESNAMES:
            counts[f"{resname} (dropped)"] = counts.get(f"{resname} (dropped)", 0) + 1
            continue

        if resname not in RESNAME_ALIASES:
            out.append(line)
            continue

        target = RESNAME_ALIASES[resname]
        counts[f"{resname} -> {target}"] = counts.get(f"{resname} -> {target}", 0) + 1

        atom_name = line[12:16]
        if resname == "MSE" and atom_name.strip() == "SE":
            atom_name = " SD "
            if len(line) > 78:
                line = line[:76] + " S" + line[78:]

        # record type -> ATOM, atom name, residue name; rest untouched
        out.append("ATOM  " + line[6:12] + atom_name + line[16] +
                   f"{target:>3s}" + line[20:])

    return "".join(out), counts


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def make_sequential_names(n: int, prefix: str = "x") -> list:
    """x001, x002, ... AF2 requires 4 lowercase alphanumeric characters."""
    if n > 999:
        raise ValueError(
            f"Too many structures ({n}) for 4-character naming (max 999). "
            f"Subsample with --step, or narrow the range with --start/--stop."
        )
    prefix = prefix[0].lower()
    if not prefix.isalnum():
        raise ValueError(f"--prefix must be alphanumeric, got {prefix!r}")
    return [f"{prefix}{i:03d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# mmCIF construction
# ---------------------------------------------------------------------------

def _three_to_one(resname: str) -> str:
    resname = RESNAME_ALIASES.get(resname.strip(), resname.strip())
    return RESTYPE_3TO1.get(resname, "X")


def _extract_chain_sequences(structure) -> dict:
    seqs = {}
    model = next(structure.get_models())
    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        seq = "".join(_three_to_one(r.get_resname()) for r in residues)
        if seq:
            seqs[chain.id] = seq
    return seqs


def _build_minimal_metadata(structure, entry_id: str) -> dict:
    chain_seqs = _extract_chain_sequences(structure)
    today = date.today().isoformat()

    unique_seqs = list(dict.fromkeys(chain_seqs.values()))
    seq_to_entity = {seq: str(i + 1) for i, seq in enumerate(unique_seqs)}

    entity_ids, seq_one_codes, seq_can_codes = [], [], []
    for seq, eid in seq_to_entity.items():
        entity_ids.append(eid)
        seq_one_codes.append(seq)
        seq_can_codes.append(seq)

    ep_entity, ep_num, ep_mon = [], [], []
    unique_resnames = set()
    struct_asym_ids, struct_asym_entity_ids = [], []
    seen_entities = set()

    model = next(structure.get_models())
    available_labels = list(string.ascii_uppercase)
    label_idx = 0

    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        seq = "".join(_three_to_one(r.get_resname()) for r in residues)
        if not seq:
            continue

        label_id = available_labels[label_idx]   # A, B, C... matching MMCIFIO
        label_idx += 1

        eid = seq_to_entity[seq]
        struct_asym_ids.append(label_id)
        struct_asym_entity_ids.append(eid)

        # one _entity_poly_seq block per entity: identical chains of a homomer
        # share an entity, and repeating the block would double its seqres.
        first_time = eid not in seen_entities
        seen_entities.add(eid)

        for pos, residue in enumerate(residues, start=1):
            resname = residue.get_resname().strip()
            unique_resnames.add(resname)
            if first_time:
                ep_entity.append(eid)
                ep_num.append(str(pos))
                ep_mon.append(resname)

    chem_comp_ids = sorted(unique_resnames)
    chem_comp_types = [
        "L-peptide linking"
        if RESNAME_ALIASES.get(r, r) in RESTYPE_3TO1
        else "other"
        for r in chem_comp_ids
    ]

    return {
        "_entry.id":                                    entry_id,
        "_struct.entry_id":                             entry_id,

        "_chem_comp.id":                                chem_comp_ids,
        "_chem_comp.type":                              chem_comp_types,

        "_struct_asym.id":                              struct_asym_ids,
        "_struct_asym.entity_id":                       struct_asym_entity_ids,

        "_entity_poly.entity_id":                       entity_ids,
        "_entity_poly.type":                            ["polypeptide(L)"] * len(entity_ids),
        "_entity_poly.pdbx_seq_one_letter_code":        seq_one_codes,
        "_entity_poly.pdbx_seq_one_letter_code_can":    seq_can_codes,

        "_entity_poly_seq.entity_id":                   ep_entity,
        "_entity_poly_seq.num":                         ep_num,
        "_entity_poly_seq.mon_id":                      ep_mon,

        "_pdbx_audit_revision_history.ordinal":         ["1"],
        "_pdbx_audit_revision_history.revision_date":   [today],
        "_pdbx_audit_revision_history.major_revision":  ["1"],
        "_pdbx_audit_revision_history.minor_revision":  ["0"],

        "_cell.length_a":                               "1.000",
        "_cell.length_b":                               "1.000",
        "_cell.length_c":                               "1.000",
        "_cell.angle_alpha":                            "90.00",
        "_cell.angle_beta":                             "90.00",
        "_cell.angle_gamma":                            "90.00",
        "_symmetry.space_group_name_H-M":               "P 1",
    }


def _write_metadata_block(f, metadata: dict):
    categories = defaultdict(dict)
    for key, val in metadata.items():
        dot_pos = key.index(".")
        categories[key[:dot_pos]][key[dot_pos + 1:]] = val

    for cat, fields in categories.items():
        is_loop = any(isinstance(v, list) for v in fields.values())
        if is_loop:
            f.write("\nloop_\n")
            field_names = list(fields.keys())
            for fn in field_names:
                f.write(f"{cat}.{fn}\n")
            n_rows = len(next(v for v in fields.values() if isinstance(v, list)))
            for row_idx in range(n_rows):
                row_vals = []
                for fn in field_names:
                    v = fields[fn]
                    val = v[row_idx] if isinstance(v, list) else v
                    if " " in str(val):
                        val = f'"{val}"'
                    row_vals.append(str(val))
                f.write(" ".join(row_vals) + "\n")
        else:
            f.write("\n")
            for fn, val in fields.items():
                if " " in str(val):
                    val = f'"{val}"'
                f.write(f"{cat}.{fn} {val}\n")


def pdb_to_cif(pdb_path: Path, cif_path: Path, af2_name: str,
               fix_resnames: bool = True) -> tuple:
    """Convert one PDB file to mmCIF with the metadata AF2 needs.

    Metadata is written before the coordinate block so AF2's parser sees
    _chem_comp before _atom_site. Returns (per-chain sequences, rename counts).
    """
    counts = {}
    source = pdb_path
    if fix_resnames:
        fixed_text, counts = normalize_pdb_text(pdb_path.read_text())
        if counts:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb",
                                             delete=False) as fh:
                fh.write(fixed_text)
                source = Path(fh.name)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(af2_name, str(source))
    if source is not pdb_path:
        source.unlink()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(tmp_path))

    coord_lines = tmp_path.read_text()
    tmp_path.unlink()
    # MMCIFIO writes "data_XXXX\n#\n" first; we write our own header
    coord_body = "\n".join(coord_lines.splitlines()[2:])

    metadata = _build_minimal_metadata(structure, af2_name)

    with open(cif_path, "w") as f:
        f.write(f"data_{af2_name}\n#\n")
        f.write("# --- AF2 required metadata (auto-generated) ---\n")
        _write_metadata_block(f, metadata)
        f.write("\n# --- Coordinates ---\n")
        f.write(coord_body)
        f.write("\n")

    return _extract_chain_sequences(structure), counts


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def frames_from_trajectory(args, tmpdir: Path, log):
    """Write selected trajectory frames as temporary PDBs. Yields paths."""
    try:
        import MDAnalysis as mda
        from MDAnalysis.exceptions import SelectionError
    except ImportError:
        sys.exit(
            "[ERROR] MDAnalysis is required for trajectory input but is not installed.\n"
            "        Do not add it to the AF2chi environment. Create a small separate\n"
            "        environment instead:\n"
            "            conda create -n af2chi-tools -c conda-forge python=3.11 mdanalysis biopython\n"
            "            conda activate af2chi-tools"
        )

    log(f"[INFO] Topology   : {args.top}")
    log(f"[INFO] Trajectory : {args.traj}")
    try:
        u = mda.Universe(args.top, args.traj)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load trajectory/topology:\n  {e}")

    n_frames = len(u.trajectory)
    log(f"[INFO] Trajectory has {n_frames} frames")

    try:
        ag = u.select_atoms(args.select)
    except SelectionError as e:
        sys.exit(f"[ERROR] Bad atom selection {args.select!r}:\n  {e}")
    if len(ag) == 0:
        sys.exit(f"[ERROR] Selection {args.select!r} matched 0 atoms.")
    log(f"[INFO] Selection {args.select!r} -> {len(ag)} atoms")

    _warn_about_chains(ag, log)

    if args.frames:
        bad = [i for i in args.frames if i < 0 or i >= n_frames]
        if bad:
            sys.exit(f"[ERROR] Frame indices out of range [0, {n_frames}): {bad}")
        indices = list(args.frames)
    else:
        start = args.start if args.start is not None else 0
        stop = args.stop if args.stop is not None else n_frames
        if start < 0 or start >= n_frames:
            sys.exit(f"[ERROR] --start {start} out of range [0, {n_frames})")
        if stop < 1 or stop > n_frames:
            sys.exit(f"[ERROR] --stop {stop} out of range [1, {n_frames}]")
        if args.step < 1:
            sys.exit("[ERROR] --step must be >= 1")
        indices = list(range(start, stop, args.step))

    if not indices:
        sys.exit("[ERROR] No frames selected - check --start / --stop / --step.")

    if args.align:
        from MDAnalysis.analysis import align as mda_align
        log("[INFO] Superimposing frames on the first selected frame (CA atoms)")
        u.trajectory[indices[0]]
        ref = u.copy()
        mda_align.AlignTraj(u, ref, select=f"({args.select}) and name CA",
                            in_memory=True).run()

    paths = []
    for n, fi in enumerate(indices, 1):
        u.trajectory[fi]
        path = tmpdir / f"frame_{fi:06d}.pdb"
        with mda.Writer(str(path), ag.n_atoms) as W:
            W.write(ag)
        paths.append(path)
    return paths


def _warn_about_chains(ag, log):
    """MD topologies often lack chain IDs, which collapses a complex into one chain."""
    try:
        chain_ids = {c.strip() for c in ag.atoms.chainIDs}
        chain_ids.discard("")
    except Exception:
        chain_ids = set()
    n_frag = len(getattr(ag, "fragments", []) or [])
    if n_frag > 1 and len(chain_ids) <= 1:
        log(
            f"[WARN] The selection has {n_frag} disconnected fragments but "
            f"{len(chain_ids) or 'no'} chain ID(s). AF2 will read this as a single "
            "chain. For a complex, set chain IDs in the topology first, e.g.\n"
            "       u.select_atoms('segid A').atoms.chainIDs = 'A'"
        )


def frames_from_pdb(args, tmpdir: Path, log):
    """Return PDB paths from a folder, a single file, or a multi-model file."""
    inp = Path(args.pdb)

    if inp.is_dir():
        paths = sorted(inp.glob("*.pdb")) + sorted(inp.glob("*.PDB"))
        if not paths:
            sys.exit(f"[ERROR] No .pdb files found in {inp}")
        log(f"[INFO] Found {len(paths)} PDB file(s) in {inp}")
        return paths

    if not inp.is_file():
        sys.exit(f"[ERROR] {inp} is not a file or directory.")

    # split a multi-model file into one PDB per model
    text = inp.read_text().splitlines(keepends=True)
    models, current = [], []
    for line in text:
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            models.append(current)
            current = []
        elif line.startswith(("ATOM", "HETATM", "TER")):
            current.append(line)
    if not models:
        log(f"[INFO] Single-model PDB: {inp}")
        return [inp]

    log(f"[INFO] Multi-model PDB: {len(models)} models in {inp}")
    paths = []
    for i, model_lines in enumerate(models):
        path = tmpdir / f"model_{i:06d}.pdb"
        path.write_text("".join(model_lines) + "END\n")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build an AF2chi template ensemble (mmCIF) from a trajectory or PDBs.",
        epilog=textwrap.dedent(__doc__),
    )

    src = p.add_argument_group("Input (choose one)")
    src.add_argument("-x", "--traj", metavar="FILE",
                     help="Trajectory file (.xtc, .trr, .dcd, ...). Requires --top.")
    src.add_argument("-t", "--top", metavar="FILE",
                     help="Topology file (.tpr, .gro, .pdb, .psf, ...).")
    src.add_argument("-p", "--pdb", metavar="PATH",
                     help="PDB file (single or multi-model) or directory of PDB files.")

    out = p.add_argument_group("Output")
    out.add_argument("-o", "--output", default="templates", metavar="DIR",
                     help="Output directory for the mmCIF ensemble [default: templates/]")
    out.add_argument("--prefix", default="x", metavar="CHAR",
                     help="Single-character prefix for output names [default: x -> x001.cif]")
    out.add_argument("--overwrite", action="store_true",
                     help="Overwrite the output directory if it already contains .cif files.")

    fr = p.add_argument_group("Frame selection (trajectory input)")
    fr.add_argument("--start", type=int, default=None, metavar="N",
                    help="First frame index, 0-based [default: first]")
    fr.add_argument("--stop", type=int, default=None, metavar="N",
                    help="Last frame index, exclusive [default: last]")
    fr.add_argument("-s", "--step", type=int, default=1, metavar="N",
                    help="Keep every N-th frame [default: 1]")
    fr.add_argument("--frames", nargs="+", type=int, default=None, metavar="N",
                    help="Explicit frame indices (overrides --start/--stop/--step)")

    sel = p.add_argument_group("Atom selection and fitting (trajectory input)")
    sel.add_argument("--select", default="protein", metavar="SELECTION",
                     help='MDAnalysis selection string [default: "protein"]')
    sel.add_argument("--align", action="store_true",
                     help="Superimpose all frames on the first selected frame (CA atoms).")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--no-resname-fix", action="store_true",
                      help="Do not rename MD residue variants (HID, CYX, MSE, ...) "
                           "to their standard equivalents. AF2 will treat them as "
                           "unknown residues.")
    misc.add_argument("-q", "--quiet", action="store_true", help="Only report errors.")

    args = p.parse_args()

    if bool(args.traj) == bool(args.pdb):
        p.error("give either -x/--traj (with -t/--top) or -p/--pdb, not both.")
    if args.traj and not args.top:
        p.error("-x/--traj requires -t/--top.")
    return args


def main():
    args = parse_args()

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.glob("*.cif")) and not args.overwrite:
        sys.exit(
            f"[ERROR] {out_dir} already contains .cif files. AF2chi would use all of "
            f"them as templates.\n        Use --overwrite, or pick an empty directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="traj2af2chi_") as td:
        tmpdir = Path(td)

        if args.traj:
            pdb_paths = frames_from_trajectory(args, tmpdir, log)
        else:
            pdb_paths = frames_from_pdb(args, tmpdir, log)

        n = len(pdb_paths)
        try:
            names = make_sequential_names(n, args.prefix)
        except ValueError as e:
            sys.exit(f"[ERROR] {e}")

        log(f"[INFO] Converting {n} structure(s) -> {out_dir}/")

        first_seqs, failed, reported_renames = None, [], False
        for i, (pdb_path, name) in enumerate(zip(pdb_paths, names), 1):
            cif_path = out_dir / f"{name}.cif"
            try:
                seqs, counts = pdb_to_cif(pdb_path, cif_path, name,
                                          fix_resnames=not args.no_resname_fix)
                if counts and not reported_renames:
                    for k, v in sorted(counts.items()):
                        log(f"[INFO] residue fix: {k}  ({v} atoms per structure)")
                    reported_renames = True
            except Exception as e:
                print(f"  [WARN] failed on {pdb_path.name}: {e}", file=sys.stderr)
                failed.append(pdb_path.name)
                continue

            if first_seqs is None:
                first_seqs = seqs
                for cid, seq in seqs.items():
                    log(f"[INFO] chain {cid!r}: {len(seq)} residues")
            elif seqs != first_seqs:
                print(
                    f"  [WARN] {name}.cif has a different sequence from the first "
                    f"structure. All ensemble members must share one sequence.",
                    file=sys.stderr,
                )

            if not args.quiet and (i % max(1, n // 10) == 0 or i == n):
                print(f"  {100 * i / n:5.1f}%  ({i}/{n})", flush=True)

    n_ok = n - len(failed)
    log(f"[DONE] Wrote {n_ok}/{n} mmCIF file(s) -> {out_dir}/")
    if failed:
        log(f"[DONE] Failed: {failed}")

    if n_ok:
        seq_len = sum(len(s) for s in (first_seqs or {}).values())
        log("")
        log("Next step:")
        log(f"  colabfold_batch --af2chi-ensemble {out_dir}/ input.fasta results/")
        if seq_len:
            log(f"  (the FASTA must contain the same {seq_len}-residue sequence)")
    if n_ok > 200:
        log(
            f"[NOTE] {n_ok} templates is a lot: the template search step scales with "
            "this number. Consider subsampling with --step."
        )


if __name__ == "__main__":
    main()

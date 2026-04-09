"""
af2sidechains_multi.py
======================
Utilities for computing sidechain chi-angle populations from AlphaFold2
predictions and generating relaxed structural ensembles.

Main components
---------------
* ``get_config``            – Build a :class:`ml_collections.ConfigDict` with all
                              runtime parameters.
* ``af2sidechain_pops``     – Compute per-residue chi-angle populations from AF2
                              angle predictions via MaxEnt reweighting.
* ``create_pdb_ensemble``   – Sample sidechain conformers from fitted populations
                              and produce an Amber-relaxed PDB ensemble.
* ``TemplateStore``         – Load mmCIF template files and expose them as
                              AlphaFold-compatible feature arrays.

Supporting functions
--------------------
* ``relax_sidechains``      – Wrapper around Amber relaxation.
* ``compute_rmsd_to_reference_biopython`` – Backbone RMSD via Biopython.
* ``count_clashes``         – Steric-clash counter using a KD-tree.
* Various small helpers: ``get_config``, ``class_to_np``, ``seq2seq``, etc.
"""

import os
import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
from scipy.optimize import minimize
from scipy.stats import circmean, circstd

from Bio import PDB
from Bio.Data import IUPACData
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Atom import Atom

from alphafold.common import residue_constants
from alphafold.common.protein import Protein, from_pdb_string, to_pdb
from alphafold.model import geometry, quat_affine, r3
from alphafold.model.all_atom import (
    atom14_to_atom37,
    atom37_to_frames,
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from alphafold.model.all_atom_multimer import (
    atom14_to_atom37 as atom14_to_atom37_multimer,
    frames_and_literature_positions_to_atom14_pos as frames_and_literature_positions_to_atom14_pos_multimer,
    torsion_angles_to_frames as torsion_angles_to_frames_multimer,
)
from alphafold.model.geometry.rigid_matrix_vector import Rigid3Array

import colabfold.relax_sc as relax
from colabfold.download import default_data_dir

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Nested dict returned by the AF2 model.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_config(
    path_rot_lib: str = None,
    path_rot_sigmas: str = None,
    path_chi_checks: str = None,
    struct_weight: float = 0.85,
    n_struct_ensemble: int = 100,
    use_gpu_relax: bool = True,
) -> ConfigDict:
    """Build and return a :class:`ml_collections.ConfigDict` with all runtime parameters.

    Parameters
    ----------
    path_rot_lib:
        Path to a custom rotamer library (.json or .csv).  When *None* the
        bundled Top8000 library is used.
    path_rot_sigmas:
        Path to a JSON file containing per-residue chi-angle standard
        deviations.  Falls back to the bundled file when *None*.
    path_chi_checks:
        Path to a CSV that lists which chi angles exist for each residue
        type.  Falls back to the bundled file when *None*.
    struct_weight:
        Weight given to the AF2 structural prior when combining it with the
        rotamer library prior (0 – 1).
    n_struct_ensemble:
        Number of structures to generate in the ensemble.

    Returns
    -------
    ConfigDict
        Fully populated configuration object.
    """
    
    config = ConfigDict()

    config.dihedral_angles = [10 * i for i in range(36)]
    config.layers = [1, 2, 3, 4, 5, 6]

    config.struct_weight = struct_weight
    config.thetas = np.geomspace(1, 10000, num=20)
    config.dihedral_angles = [5 + (10 * i) for i in range(36)]
    config.n_struct_ensemble = int(n_struct_ensemble)
    config.pool_cpus = max(os.cpu_count(), int(os.cpu_count() / 4))

    #set default af2chi param dir
    default_af2chi_param_dir = os.path.join(default_data_dir, "af2chi-params")

    ## setup gpu/cpu relaxation
    config.use_gpu_relax = use_gpu_relax

    #if path_rot_lib != None:
    #    config.rot_lib = pd.read_csv(path_rot_lib, index_col=0)
    if path_rot_lib is not None:
        if path_rot_lib.endswith(".json"):
            with open(path_rot_lib) as f:
                config.rot_lib = json.load(f)
        else:
            config.rot_lib = pd.read_csv(path_rot_lib, index_col=0)
        
        config.top8000 = pd.read_csv(
                os.path.join(default_af2chi_param_dir,"Top8000_rebinned_all_chi_distributions.csv"),
                index_col=0)
        config.custom_rotamer_lib = True
    else:
        config.rot_lib = pd.read_csv(
                os.path.join(default_af2chi_param_dir,"Top8000_rebinned_all_chi_distributions.csv"),
            index_col=0,
        )
        config.custom_rotamer_lib = False

    if path_rot_sigmas != None:
        with open(path_rot_sigmas) as f:
            config.sigmas_restype = json.load(f)
    else:
        with open(
            os.path.join(default_af2chi_param_dir,"Top8000_all_chi_sigmas.json")
        ) as f:
            config.sigmas_restype = json.load(f)

    if path_chi_checks != None:
        config.res_has_chi = pd.read_csv(path_chi_checks, index_col=0)
    else:
        config.res_has_chi = pd.read_csv(
            os.path.join(default_af2chi_param_dir,"res_chis.csv"),
            index_col=0,
        )

    config.dict_chi2layer = {
        "chi1": 3,
        "chi2": 4,
    }

    return config

# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------

def class_to_np(c):
    """Return a copy of *c* with all JAX arrays converted to NumPy arrays.

    Parameters
    ----------
    c:
        An object whose ``__dict__`` may contain :class:`jax.numpy.ndarray`
        values (possibly nested inside plain dicts).

    Returns
    -------
    object
        A new lightweight object with the same attributes, but every
        ``jnp.ndarray`` replaced by a ``np.ndarray``.
    """
    class _Dict2Obj:
        def __init__(self, d):
            for k, v in _jnp_to_np(d).items():
                setattr(self, k, v)

    return _Dict2Obj(c.__dict__)


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert JAX arrays to NumPy arrays inside a dict.

    Parameters
    ----------
    output:
        A (possibly nested) dictionary that may contain
        :class:`jax.numpy.ndarray` leaves.

    Returns
    -------
    dict
        The same dictionary with all ``jnp.ndarray`` values replaced by
        ``np.ndarray``.
    """
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def slice_batch(batch: dict, i: int) -> dict:
    """Return the *i*-th element of every array in *batch*.

    Parameters
    ----------
    batch:
        Dictionary mapping string keys to arrays whose first axis indexes
        batch elements.
    i:
        Batch index to select.

    Returns
    -------
    dict
        Dictionary with the same keys, each value being ``batch[k][i]``.
    """
    return {k: v[i] for k, v in batch.items()}


def seq2seq(sequence: str) -> list:
    """Convert a one-letter amino-acid sequence to a list of three-letter residue IDs.

    Parameters
    ----------
    sequence:
        One-letter amino-acid sequence string (e.g. ``"ACDEF"``).

    Returns
    -------
    list of str
        Three-letter residue identifiers with 1-based position appended,
        e.g. ``["ALA1", "CYS2", "ASP3", ...]``.
    """
    return [
        f"{IUPACData.protein_letters_1to3[aa].upper()}{i + 1}"
        for i, aa in enumerate(sequence)
    ]


# ---------------------------------------------------------------------------
# OpenMM patch
# ---------------------------------------------------------------------------

def patch_openmm():
    """Apply the AlphaFold disulphide-bond patch to OpenMM 7.7.1.

    This replicates the fix from the official AlphaFold Docker patch
    (licensed CC-0) targeting OpenMM PR #3203.  It replaces
    ``app.Topology.createDisulfideBonds`` with a corrected implementation
    that prevents a single atom from participating in multiple disulphide
    bonds.
    """
    from openmm import app
    from openmm.unit import nanometers, sqrt

    def createDisulfideBonds(self, positions):
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return "SG" in names and "HG" not in names

        def isDisulfideBonded(atom):
            for b in self._bonds:
                if atom in b and b[0].name == "SG" and b[1].name == "SG":
                    return True
            return False

        cyx = [res for res in self.residues() if res.name == "CYS" and isCyx(res)]
        atomNames = [[atom.name for atom in res._atoms] for res in cyx]

        for i in range(len(cyx)):
            sg1 = cyx[i]._atoms[atomNames[i].index("SG")]
            pos1 = positions[sg1.index]
            candidate_distance, candidate_atom = 0.3 * nanometers, None

            for j in range(i):
                sg2 = cyx[j]._atoms[atomNames[j].index("SG")]
                pos2 = positions[sg2.index]
                delta = [x - y for (x, y) in zip(pos1, pos2)]
                distance = sqrt(
                    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
                )
                if distance < candidate_distance and not isDisulfideBonded(sg2):
                    candidate_distance = distance
                    candidate_atom = sg2

            if candidate_atom:
                self.addBond(sg1, candidate_atom)

    app.Topology.createDisulfideBonds = createDisulfideBonds


# ---------------------------------------------------------------------------
# Amber relaxation wrapper
# ---------------------------------------------------------------------------

def relax_sidechains(
    pdb_filename=None,
    sampled_angles=None,
    config=None,
    pdb_lines=None,
    pdb_obj=None,
    use_gpu: bool = False,
    max_iterations: int = 0,
    stiffness: float = 1.0,
):
    """Run Amber sidechain relaxation on a protein structure.

    Exactly one of *pdb_filename*, *pdb_lines*, or *pdb_obj* must be
    supplied; they are tried in that order of priority.

    Parameters
    ----------
    pdb_filename:
        Path to a PDB file on disk.
    sampled_angles:
        Sampled torsion angles forwarded to the custom Amber relaxer.
    config:
        Runtime configuration (forwarded to the relaxer).
    pdb_lines:
        PDB file content as a string.
    pdb_obj:
        Pre-parsed :class:`alphafold.common.protein.Protein` object.
    use_gpu:
        Whether to run OpenMM on the GPU.
    max_iterations:
        Maximum number of Amber minimisation steps (0 = until convergence).
    stiffness:
        Force-constant (kcal/mol/Å²) applied to positional restraints.

    Returns
    -------
    str
        Relaxed structure as a PDB-format string.
    """
    if "relax" not in dir():
        patch_openmm()

    if pdb_obj is None:
        if pdb_lines is None:
            pdb_lines = Path(pdb_filename).read_text()
        pdb_obj = from_pdb_string(pdb_lines)

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=max_iterations,
        tolerance=2.39,
        stiffness=stiffness,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu,
        sampled_angles=sampled_angles,
        config=config,
    )

    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)
    return relaxed_pdb_lines


# ---------------------------------------------------------------------------
# Chi-angle population calculator
# ---------------------------------------------------------------------------

class af2sidechain_pops:
    """Compute per-residue chi-angle populations from AlphaFold2 predictions.

    The class uses a MaxEnt reweighting scheme to fit discrete rotamer
    populations so that the ensemble-average chi angle is consistent with
    the mean predicted by each of AF2's internal recycling layers.

    Parameters
    ----------
    config:
        Configuration object returned by :func:`get_config`.
    is_complex:
        Set to *True* when the target is a multimeric complex; selects the
        multimer code path for angle extraction.
    """

    def __init__(self, config, is_complex: bool = False):
        self.config = config
        self.is_complex = is_complex

    def __call__(self, sequence: list, angles: np.ndarray, chi_sel: str):
        """Compute chi populations for all residues that have *chi_sel*.

        Parameters
        ----------
        sequence:
            List of one-letter amino-acid codes for each chain.
        angles:
            Predicted torsion angles from AF2, shape ``(layers, N, 7, 2)``.
        chi_sel:
            Which chi angle to process (``"chi1"``, ``"chi2"``, …).

        Returns
        -------
        tuple[dict, dict]
            ``(fitted_pops, prior_pops)`` — both are dicts mapping
            three-letter residue IDs (e.g. ``"ALA5"``) to lists of bin
            probabilities.
        """

        chi_internal_layers_mean, chi_internal_layers_std, chi_final_layer = (
            self.af2chi(angles, self.config.dict_chi2layer[chi_sel])
        )

        fitted_pops, prior_pops = self.eval_chi_populations(
            seq2seq("".join(sequence)),
            chi_sel,
            chi_internal_layers_mean,
            chi_internal_layers_std,
            chi_final_layer,
        )

        return fitted_pops, prior_pops

    # ------------------------------------------------------------------
    # Internal angle-extraction methods
    # ------------------------------------------------------------------

    def af2chi(
        self, angles: np.ndarray, layer_chi: int
    ) -> tuple:
        """Extract per-layer circular statistics for a given chi angle.

        Parameters
        ----------
        angles:
            AF2 predicted angles, shape ``(layers, N, 7, 2)`` where the
            last two dimensions represent (sin, cos).
        layer_chi:
            Index of the chi angle within the AF2 internal representation
            (e.g. 3 for chi1).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(mean, std, final_layer_values)`` – circular mean and standard
            deviation across the internal recycling layers, plus the raw
            angles from the final layer.
        """
        chi_internal_layers = self.extract_chi_angles(
            angles, layer_chi, self.config.layers, is_last=False
        ).astype(np.float64)

        chi_internal_layers_mean = circmean(
            chi_internal_layers, low=0, high=2 * np.pi, axis=0
        )
        chi_internal_layers_std = circstd(
            chi_internal_layers, low=0, high=2 * np.pi, axis=0
        )
        chi_final_layer = self.extract_chi_angles(
            angles, layer_chi, is_last=True
        ).flatten()

        return chi_internal_layers_mean, chi_internal_layers_std, chi_final_layer

    def extract_chi_angles(
        self,
        angles: np.ndarray,
        layer_chi: int,
        layers: list = [],
        is_last: bool = False,
    ) -> np.ndarray:
        """Convert AF2 (sin, cos) angle pairs to circular-mean chi values.

        Parameters
        ----------
        angles:
            AF2 predicted angles, shape ``(layers, N, 7, 2)``.
        layer_chi:
            Chi-angle index within the angle tensor.
        layers:
            List of recycling-layer indices to extract from.  Ignored when
            *is_last* is *True*.
        is_last:
            When *True* only the final layer (index ``-1``) is extracted.

        Returns
        -------
        np.ndarray
            Chi angles in radians, shape ``(len(layers), N)``.
        """
        if is_last:
            layers = [-1]

        chi_angles = []
        for layer in layers:
            raw = np.arctan2(
                angles[layer][:, layer_chi, 0],
                angles[layer][:, layer_chi, 1],
            ).reshape(-1, 1)
            chi_angles.append(
                circmean(raw, low=0, high=2 * np.pi, axis=1)
            )

        return np.array(chi_angles)

    # ------------------------------------------------------------------
    # Population-fitting pipeline
    # ------------------------------------------------------------------

    def eval_chi_populations(
        self,
        sequence: list,
        chi_sel: str,
        chi_internal_layers_mean: np.ndarray,
        chi_internal_layers_std: np.ndarray,
        chi_final_layer: np.ndarray,
    ) -> dict:
        """Fit chi populations for all residues with the selected chi angle.

        Parameters
        ----------
        sequence:
            List of three-letter residue IDs (e.g. ``["ALA1", "CYS2", …]``).
        chi_sel:
            Chi angle label (e.g. ``"chi1"``).
        chi_internal_layers_mean:
            Circular mean over internal layers, shape ``(N,)``.
        chi_internal_layers_std:
            Circular std over internal layers, shape ``(N,)``.
        chi_final_layer:
            Chi values from the final AF2 layer, shape ``(N,)``.

        Returns
        -------
        tuple[dict, dict]
            ``(fitted_pops_residues, prior_pops_residues)`` keyed by
            three-letter residue IDs.
        """
        res_list = self.res_list_with_sel_chi(sequence, chi_sel)

        if self.config.custom_rotamer_lib:
            print("Using custom rotamer library")
            prior_comb = self.create_custom_combined_prior(
                res_list,
                chi_sel,
                chi_final_layer,
                sigmas_restype=self.config.sigmas_restype,
                structure_weight=0.1,
                rot_lib=self.config.rot_lib,
                dihedral_angles=self.config.dihedral_angles,
            )
        else:
            prior_comb = self.create_combined_prior(
                res_list,
                chi_sel,
                chi_final_layer,
                sigmas_restype=self.config.sigmas_restype,
                structure_weight=self.config.struct_weight,
                rot_lib=self.config.rot_lib,
                dihedral_angles=self.config.dihedral_angles,
            )

        fitted_pops_residues, prior_pops_residues = {}, {}

        pool = Pool(self.config.pool_cpus)
        results = pool.starmap(
            self.maxent_rw_populations,
            [
                (res, chi_internal_layers_mean, chi_internal_layers_std, prior_comb, self.config)
                for res in res_list
            ],
        )
        pool.close()
        pool.join()

        for res, fitted_pops, prior_pops in results:
            fitted_pops_residues[res] = list(fitted_pops)
            prior_pops_residues[res] = list(prior_pops)

        return fitted_pops_residues, prior_pops_residues

    # ------------------------------------------------------------------
    # Prior construction
    # ------------------------------------------------------------------

    def create_custom_combined_prior(
        self,
        sequence: list,
        chi_sel: str,
        chi_final_layer: np.ndarray,
        sigmas_restype: dict,
        structure_weight: float,
        rot_lib: dict,
        dihedral_angles: list,
    ) -> dict:
        """Build a combined prior from a custom rotamer library and the AF2 structure.

        Parameters
        ----------
        sequence:
            Three-letter residue ID list.
        chi_sel:
            Chi angle label.
        chi_final_layer:
            AF2 final-layer chi values, shape ``(N,)``.
        sigmas_restype:
            Per-residue chi standard deviations.
        structure_weight:
            Weight for the structural (Gaussian) component (0–1).
        rot_lib:
            Custom rotamer library keyed by chi label then residue ID.
        dihedral_angles:
            Bin centres (degrees) for the discrete distribution.

        Returns
        -------
        dict
            Normalised prior distribution for each residue in *sequence*.
        """
        def gaussian(x, a, mu, sigma):
            return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(
                -0.5 * (np.square(x - mu) / np.square(sigma))
            )

        prior_combined = {}
        for residue in sequence:
            restype = residue[:3]
            chi_final_layer_deg = np.rad2deg(
                circmean(
                    chi_final_layer[int(residue[3:]) - 1].reshape(-1, 1),
                    low=0,
                    high=2 * np.pi,
                    axis=1,
                )
            )
            w0_structure = gaussian(
                dihedral_angles, 1, chi_final_layer_deg, sigmas_restype[restype + "_" + chi_sel]
            )
            w0_structure /= np.sum(w0_structure)
            w0_rot_lib = np.array(rot_lib[chi_sel][residue])
            w0_comb = structure_weight * w0_structure + (1 - structure_weight) * w0_rot_lib
            w0_comb /= np.sum(w0_comb)
            prior_combined[residue] = w0_comb

        return prior_combined

    def create_custom_combined_top8000_prior(
        self,
        sequence: list,
        chi_sel: str,
        chi_final_layer: np.ndarray,
        sigmas_restype: dict,
        structure_weight: float,
        rot_lib: dict,
        dihedral_angles: list,
        top8000: dict,
    ) -> dict:
        """Build a combined prior mixing a custom library with the Top8000 database.

        Parameters
        ----------
        sequence:
            Three-letter residue ID list.
        chi_sel:
            Chi angle label.
        chi_final_layer:
            AF2 final-layer chi values, shape ``(N,)``.
        sigmas_restype:
            Per-residue chi standard deviations (unused in this variant).
        structure_weight:
            Weight for the Top8000 component (0–1).
        rot_lib:
            Custom rotamer library keyed by chi label then residue ID.
        dihedral_angles:
            Bin centres (degrees) for the discrete distribution.
        top8000:
            Top8000 reference distributions keyed by ``"<RESTYPE>_<chi>"``.

        Returns
        -------
        dict
            Normalised prior distribution for each residue in *sequence*.
        """
        prior_combined = {}
        for residue in sequence:
            restype = residue[:3]
            w0_rot_lib = np.array(rot_lib[chi_sel][residue])
            w0_top8000 = np.array(top8000[restype + "_" + chi_sel])
            w0_top8000 /= np.sum(w0_top8000)
            w0_comb = structure_weight * w0_top8000 + (1 - structure_weight) * w0_rot_lib
            w0_comb /= np.sum(w0_comb)
            prior_combined[residue] = w0_comb

        return prior_combined

    def create_combined_prior(
        self,
        sequence: list,
        chi_sel: str,
        chi_final_layer: np.ndarray,
        sigmas_restype: dict,
        structure_weight: float,
        rot_lib: dict,
        dihedral_angles: list,
    ) -> dict:
        """Build a combined prior from the Top8000 rotamer library and the AF2 structure.

        Parameters
        ----------
        sequence:
            Three-letter residue ID list.
        chi_sel:
            Chi angle label.
        chi_final_layer:
            AF2 final-layer chi values, shape ``(N,)``.
        sigmas_restype:
            Per-residue chi standard deviations.
        structure_weight:
            Weight for the structural (Gaussian) component (0–1).
        rot_lib:
            Top8000 rotamer library keyed by ``"<RESTYPE>_<chi>"``.
        dihedral_angles:
            Bin centres (degrees) for the discrete distribution.

        Returns
        -------
        dict
            Normalised prior distribution for each residue in *sequence*.
        """
        def gaussian(x, a, mu, sigma):
            return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(
                -0.5 * (np.square(x - mu) / np.square(sigma))
            )

        prior_combined = {}
        for residue in sequence:
            restype = residue[:3]
            chi_final_layer_deg = np.rad2deg(
                circmean(
                    chi_final_layer[int(residue[3:]) - 1].reshape(-1, 1),
                    low=0,
                    high=2 * np.pi,
                    axis=1,
                )
            )
            w0_structure = gaussian(
                dihedral_angles, 1, chi_final_layer_deg, sigmas_restype[restype + "_" + chi_sel]
            )
            w0_structure /= np.sum(w0_structure)
            w0_rot_lib = np.array(rot_lib[restype + "_" + chi_sel])
            w0_rot_lib /= np.sum(w0_rot_lib)
            w0_comb = structure_weight * w0_structure + (1 - structure_weight) * w0_rot_lib
            w0_comb /= np.sum(w0_comb)
            prior_combined[residue] = w0_comb

        return prior_combined

    # ------------------------------------------------------------------
    # Residue filtering
    # ------------------------------------------------------------------

    def res_list_with_sel_chi(self, sequence: list, chi_sel: str) -> list:
        """Return the subset of residues that possess *chi_sel*.

        Parameters
        ----------
        sequence:
            Three-letter residue ID list.
        chi_sel:
            Chi angle label (e.g. ``"chi2"``).

        Returns
        -------
        list of str
            Residue IDs from *sequence* whose residue type has *chi_sel*.
        """
        return [
            res
            for res in sequence
            if self.config.res_has_chi.query(f'restype == "{res[:3]}"')[chi_sel].values[0]
        ]

    # ------------------------------------------------------------------
    # MaxEnt reweighting
    # ------------------------------------------------------------------

    def maxent_rw_populations(
        self,
        res: str,
        chi_internal_layers_mean: np.ndarray,
        chi_internal_layers_std: np.ndarray,
        prior_combined: dict,
        config: ml_collections.ConfigDict,
    ) -> list:
        """Fit a discrete rotamer distribution using MaxEnt reweighting.

        The optimisation finds the population vector *w* that minimises the
        relative entropy with respect to *prior_combined[res]* while
        satisfying ``chi2 ≤ 1`` against the AF2-predicted mean and
        uncertainty.

        Parameters
        ----------
        res:
            Three-letter residue ID (e.g. ``"SER42"``).
        chi_internal_layers_mean:
            Circular mean over internal layers for all residues, shape ``(N,)``.
        chi_internal_layers_std:
            Circular std over internal layers for all residues, shape ``(N,)``.
        prior_combined:
            Combined prior distributions keyed by residue ID.
        config:
            Runtime configuration (provides ``thetas``, ``dihedral_angles``).

        Returns
        -------
        list
            ``[res, fitted_populations, prior_populations]``
        """
        warnings.filterwarnings("ignore")

        # ----- nested helper functions ------------------------------------

        def maxent_loss(w1, w0, dihedral_angles, dihedral_avg_AF, dihedral_avg_AF_err, theta):
            """MaxEnt objective: 0.5 * chi2 - theta * S(w1 || w0), normalised by theta."""
            dihedral_avg_calc = np.rad2deg(
                circweightedmean(np.deg2rad(dihedral_angles), weights=w1)
            )
            chisquare = eval_chisquare(dihedral_avg_calc, dihedral_avg_AF, dihedral_avg_AF_err)
            L = 0.5 * chisquare - theta * (-np.sum(w1 * np.log(w1 / w0)))
            return L / theta

        def eval_chisquare(angle1: float, angle2: float, angle_err: float) -> float:
            """Chi-squared statistic for the difference between two angles."""
            return np.square(
                diff_angles(np.deg2rad(angle1), np.deg2rad(angle2)) / np.deg2rad(angle_err)
            )

        def theta_loc(thetas, chisquare):
            """Select the regularisation parameter theta.

            Chooses the largest theta with chi2 ≤ 1; falls back to the
            theta that minimises chi2 when none satisfy that criterion.
            """
            if np.any(chisquare <= 1.0):
                idx = np.argwhere(chisquare <= 1.0).flatten()[-1]
            else:
                idx = np.argmin(chisquare)
            return thetas[idx], idx

        def diff_angles(angle1: float, angle2: float) -> float:
            """Signed circular difference between two angles (radians)."""
            return np.angle(np.exp(1j * (angle1 - angle2)))

        def circweightedmean(angles, weights=None):
            """Circular weighted mean of *angles* (radians)."""
            if weights is None:
                weights = np.ones(len(angles))
            weights_norm = weights / np.sum(weights)
            if np.all(weights_norm == weights_norm[0]):
                sin_avg, cos_avg = 0.0, 0.0
            else:
                sin_avg = np.sum(weights_norm * np.sin(angles))
                cos_avg = np.sum(weights_norm * np.cos(angles))
            return np.arctan2(sin_avg, cos_avg)

        # ----- residue-level setup ----------------------------------------

        resnum = int(res[3:])
        chi_avg_target = np.rad2deg(chi_internal_layers_mean[resnum - 1])
        chi_err_target = 2.0 * float(np.rad2deg(chi_internal_layers_std[resnum - 1]))

        if np.any(prior_combined[res] == np.nan):
            return [res, prior_combined[res], prior_combined[res]]

        w0 = prior_combined[res]
        # Weights cannot be exactly zero in MaxEnt.
        w0[w0 == 0.0] = 1e-32

        bnds = ((1e-32, 1),) * len(w0)
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)

        # ----- theta sweep ------------------------------------------------

        fitted_pops_vs_theta = []
        chisquare_vs_theta = np.zeros(len(config.thetas))

        for j, theta in enumerate(config.thetas):
            args = (w0, config.dihedral_angles, chi_avg_target, chi_err_target, theta)
            min_result = minimize(
                maxent_loss,
                w0,
                args=args,
                bounds=bnds,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-12},
            )
            weights_fit = min_result["x"]
            weights_fit /= np.sum(weights_fit)

            chisquare_vs_theta[j] = eval_chisquare(
                np.rad2deg(
                    circweightedmean(np.deg2rad(config.dihedral_angles), weights=weights_fit)
                ),
                chi_avg_target,
                chi_err_target,
            )
            fitted_pops_vs_theta.append(weights_fit)

        theta_sel, idx = theta_loc(config.thetas, chisquare_vs_theta)
        return [res, np.array(fitted_pops_vs_theta[idx]), np.array(w0)]


# ---------------------------------------------------------------------------
# PDB ensemble generator
# ---------------------------------------------------------------------------

class create_pdb_ensemble:
    """Generate a relaxed PDB ensemble by sampling sidechain conformers.

    For each ensemble member the method:

    1. Randomly selects a backbone from the pool (if multiple templates are
       available).
    2. Samples chi1/chi2 angles from the fitted population distributions.
    3. Builds atomic coordinates via AlphaFold's frame-based reconstruction.
    4. Runs Amber relaxation.
    5. Accepts the structure if it passes clash and RMSD filters.

    Parameters
    ----------
    config:
        Runtime configuration returned by :func:`get_config`.
    results:
        Dictionary containing ``"input_features"``, ``"sc_atom_mask"``,
        ``"template_all_atom_mask"``, and ``"template_aatype"`` as produced
        by the AF2 pipeline.
    result_dir:
        :class:`~pathlib.Path` to the output directory.  A sub-directory
        ``sidechain_ensemble`` will be created inside it.
    ref_pdb_path:
        Path to the reference PDB used for RMSD filtering when *use_ref* is
        *True*.
    is_complex:
        Select the multimer code path when *True*.
    stiffness:
        Force constant (kcal/mol/Å²) for Amber positional restraints.
    use_ref:
        When *True* RMSD is computed against *ref_pdb_path*; when *False*
        RMSD is computed between the non-relaxed and relaxed versions.
    """

    def __init__(
        self,
        config,
        results,
        result_dir,
        ref_pdb_path,
        is_complex: bool = False,
        stiffness: float = 1.0,
        use_ref: bool = True,
    ):
        self.config = config
        self.pdb_input_features = results["input_features"]
        self.mask_atom = results["sc_atom_mask"]
        self.result_dir = result_dir.joinpath("sidechain_ensemble")
        self.sampler = np.random.default_rng()
        self.reference_pdb_path = ref_pdb_path
        self.is_complex = is_complex
        self.stiffness = stiffness
        self.use_ref = use_ref

    def __call__(
        self,
        sequence: list,
        backbone: jnp.ndarray,
        angles: jnp.ndarray,
        fitted_pops: dict,
        jobname: str,
    ) -> list:
        """Generate the ensemble and write relaxed PDB files to disk.

        Parameters
        ----------
        sequence:
            List of one-letter amino-acid codes for each chain.
        backbone:
            Predicted backbone frames, shape compatible with
            :meth:`backbone_r3_creation`.
        angles:
            Predicted torsion angles from AF2, shape ``(layers, N, 7, 2)``.
        fitted_pops:
            Per-chi, per-residue population distributions as returned by
            :class:`af2sidechain_pops`.
        jobname:
            Base name used to construct output PDB file names.

        Returns
        -------
        list of bool
            One entry per attempted structure; *True* means accepted.
        """
        r3_bb_list, aatype_list = self.backbone_r3_creation(backbone, self.is_complex)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        angles = angles[-1, ...]
        ensemble = 0
        production_pool = []
        max_size_try = 3 * self.config.n_struct_ensemble

        print(f"Generating ensemble of {self.config.n_struct_ensemble} structures")

        while ensemble < self.config.n_struct_ensemble:
            if len(r3_bb_list) > 1:
                rng = self.sampler.integers(0, len(r3_bb_list), endpoint=False)
                r3_bb = r3_bb_list[rng]
                aatype = aatype_list[rng]
            else:
                r3_bb = r3_bb_list[0]
                aatype = aatype_list[0]

            print(f">>> Generating ensemble structure {ensemble} ...")
            production_pool.append(
                self.ensemble_creation_iteration(
                    sequence, aatype, r3_bb, angles, fitted_pops, jobname, ensemble, self.use_ref
                )
            )
            if production_pool[-1]:
                ensemble += 1
            if len(production_pool) > max_size_try:
                print("max try for ensemble generation reached, breaking")
                break

        return production_pool

    # ------------------------------------------------------------------
    # Backbone → rigid-body frames
    # ------------------------------------------------------------------

    def backbone_r3_creation(
        self,
        backbone: np.ndarray,
        is_complex: bool = False,
    ) -> tuple:
        """Convert backbone coordinate arrays to AF2 rigid-body frame objects.

        Accepts backbones in five formats:

        * ``(N, 7)``       – quaternion + translation (monomer, single template).
        * ``(T, N, 7)``    – multiple templates, quaternion form (monomer).
        * ``(N, 3, 4)``    – rotation matrix + translation (multimer, single
                             template).  This is the compact rigid-frame format
                             produced by AF2-multimer's structure module:
                             ``backbone[:, :3, :3]`` is the 3×3 rotation matrix
                             and ``backbone[:, :, 3]`` is the translation vector.
        * ``(T, N, 3, 4)`` – multiple templates of the same format.
        * ``(T, N, atoms, 3)`` – raw Cartesian atom coordinates; N, CA, C
                                  (indices 0, 1, 2) are used to compute the
                                  local frame via Gram-Schmidt.

        Parameters
        ----------
        backbone:
            Backbone representation (see above).
        is_complex:
            When *True* constructs :class:`geometry.Rigid3Array` objects
            (multimer path); otherwise constructs :class:`r3.Rigids`.

        Returns
        -------
        tuple[list, list]
            ``(r3_list, aatype_list)`` — one rigid-body object and one
            aatype array per template.
        """
        from alphafold.model.geometry import rigid_matrix_vector as geom_rigid
        from alphafold.model.geometry import rotation_matrix as geom_rot
        from alphafold.model.geometry import vector as geom_vec

        backbone = np.array(backbone)

        # ------------------------------------------------------------------
        # Normalise to at least 3-D so every branch below sees (T, N, ...).
        # A 2-D array is the single-template monomer case (N, 7).
        # A 3-D array with last two dims (3, 4) is the single-template
        # multimer compact format (N, 3, 4) — lift to (1, N, 3, 4) here
        # so it is handled identically to the multi-template (T, N, 3, 4)
        # case in the 4-D branch below.
        # ------------------------------------------------------------------
        if backbone.ndim == 2:
            backbone = backbone[None, ...]          # (N, 7)    → (1, N, 7)
        elif backbone.ndim == 3 and backbone.shape[-2:] == (3, 4):
            backbone = backbone[None, ...]          # (N, 3, 4) → (1, N, 3, 4)

        backbone_4d_complex = None   # sentinel used in the per-template loop

        # ------------------------------------------------------------------ #
        # 4-D input — two sub-cases disambiguated by the last-two-dim shape:  #
        #   (T, N, 3, 4)    → multimer compact rotation+translation           #
        #   (T, N, atoms, 3) → raw Cartesian, need Gram-Schmidt frames        #
        # ------------------------------------------------------------------ #
        if backbone.ndim == 4 and backbone.shape[-2:] == (3, 4):
            # (T, N, 3, 4): AF2-multimer compact rigid-frame format.
            # backbone[t, n, :3, :3] = 3×3 rotation matrix (rows = basis vecs)
            # backbone[t, n, :,  3]  = translation vector
            rot_mat     = backbone[:, :, :3, :3]  # (T, N, 3, 3)
            translation = backbone[:, :, :, 3]    # (T, N, 3)
            backbone_4d_complex = (rot_mat, translation)
            backbone = None   # consumed

        elif backbone.ndim == 4:
            # (T, N, n_atoms, 3): raw Cartesian atom coordinates.
            # Derive rigid frames from N (0), CA (1), C (2) via
            # make_transform_from_reference (Gram-Schmidt).
            n_xyz  = backbone[:, :, 0, :]  # (T, N, 3)
            ca_xyz = backbone[:, :, 1, :]  # (T, N, 3)
            c_xyz  = backbone[:, :, 2, :]  # (T, N, 3)

            num_templates, n_res = backbone.shape[:2]

            rotation_flat, translation_flat = quat_affine.make_transform_from_reference(
                n_xyz=n_xyz.reshape(-1, 3),
                ca_xyz=ca_xyz.reshape(-1, 3),
                c_xyz=c_xyz.reshape(-1, 3),
            )

            if is_complex:
                rotation_mat = rotation_flat.reshape(num_templates, n_res, 3, 3)
                translation  = translation_flat.reshape(num_templates, n_res, 3)
                backbone_4d_complex = (rotation_mat, translation)
                backbone = None
            else:
                quat_flat  = quat_affine.rot_to_quat(rotation_flat, unstack_inputs=True)
                backbone_7 = jnp.concatenate([quat_flat, translation_flat], axis=-1)
                backbone   = backbone_7.reshape(num_templates, n_res, 7)

        # ------------------------------------------------------------------ #
        # 3-D input: (T, N, 7) — quaternion + translation, monomer only.     #
        # ------------------------------------------------------------------ #
        elif backbone.ndim == 3:
            pass  # backbone already correct; backbone_4d_complex stays None

        else:
            raise ValueError(
                f"Backbone must have 2, 3, or 4 dimensions, got {backbone.ndim}"
            )

        # ------------------------------------------------------------------ #
        # Per-template loop                                                    #
        # ------------------------------------------------------------------ #
        r3_list     = []
        aatype_list = []

        num_templates = (
            backbone_4d_complex[0].shape[0]
            if backbone_4d_complex is not None
            else backbone.shape[0]
        )

        for i in range(num_templates):

            if is_complex:
                if backbone_4d_complex is not None:
                    rot_i   = jnp.array(backbone_4d_complex[0][i], dtype=jnp.float32)  # (N, 3, 3)
                    trans_i = jnp.array(backbone_4d_complex[1][i], dtype=jnp.float32)  # (N, 3)
                    rot3 = geom_rot.Rot3Array.from_array(rot_i)
                    vec3 = geom_vec.Vec3Array(
                        x=trans_i[..., 0],
                        y=trans_i[..., 1],
                        z=trans_i[..., 2],
                    )
                    r3_bb = geom_rigid.Rigid3Array(rotation=rot3, translation=vec3)
                else:
                    # backbone is (T, N, 7) but is_complex=True
                    r3_bb = geom_rigid.Rigid3Array.from_array(
                        jnp.array(backbone[i]).astype(jnp.float32)
                    )
                aatype = jnp.array(self.pdb_input_features["aatype"])

            else:
                # Monomer path
                quat_bb = quat_affine.QuatAffine.from_tensor(jnp.array(backbone[i]))
                r3_bb   = r3.rigids_from_quataffine(quat_bb)

                pdb_aatype = self.pdb_input_features.get("aatype")
                if pdb_aatype is None:
                    aatype = jnp.array([])
                elif isinstance(pdb_aatype, (list, tuple, np.ndarray)) and len(pdb_aatype) > i:
                    aatype = jnp.array(pdb_aatype[i])
                else:
                    aatype = jnp.array(pdb_aatype[0])

            r3_list.append(r3_bb)
            aatype_list.append(aatype)

        return r3_list, aatype_list

    def ensemble_creation_iteration(
        self,
        sequence: list,
        aatype: np.ndarray,
        r3_bb: jnp.ndarray,
        angles: jnp.ndarray,
        fitted_pops: dict,
        jobname: str,
        ensemble: int,
        use_ref: bool,
    ) -> bool:
        """Generate, relax, and validate a single ensemble member.

        Parameters
        ----------
        sequence:
            List of one-letter amino-acid codes.
        aatype:
            Integer amino-acid type array, shape ``(N,)``.
        r3_bb:
            Backbone rigid-body frames for this template.
        angles:
            Final-layer torsion angles, shape ``(N, 7, 2)``.
        fitted_pops:
            Per-chi population distributions.
        jobname:
            Base name for output files.
        ensemble:
            Index of this ensemble member (used in file names).
        use_ref:
            When *True* RMSD is computed against the reference structure.

        Returns
        -------
        bool
            *True* if the structure was accepted (no clashes, RMSD within
            threshold); *False* otherwise.
        """
        sampled_angles = self.sample_sidechains(sequence, angles, fitted_pops, self.sampler)
        jnp_sampled_angles = jnp.array(sampled_angles)

        if self.is_complex:
            structure = self.generate_structure_multimer(aatype, r3_bb, jnp_sampled_angles)
        else:
            structure = self.generate_structure(aatype, r3_bb, jnp_sampled_angles)

        pdb_lines = to_pdb(class_to_np(structure))

        unrelaxed_path = self.result_dir.joinpath(
            jobname + f"_sidechain_ensemble_{ensemble}.pdb"
        )
        relaxed_path = self.result_dir.joinpath(
            jobname + f"_sidechain_ensemble_relaxed_{ensemble}.pdb"
        )

        Path(unrelaxed_path).write_text(pdb_lines)
        check_clashes = count_clashes(unrelaxed_path)
        print(f"non-relaxed structure clashes: {check_clashes}")

        relaxed_pdb_lines = relax_sidechains(
            pdb_lines=pdb_lines,
            sampled_angles=sampled_angles,
            config=self.config,
            use_gpu=self.config.use_gpu_relax,
            max_iterations=0,
            stiffness=self.stiffness,
        )
        Path(relaxed_path).write_text(relaxed_pdb_lines)

        check_clashes = count_clashes(relaxed_path)

        if use_ref:
            print("Computing RMSD to input template structure")
            rmsd = compute_rmsd_to_reference_biopython(
                self.reference_pdb_path, relaxed_path.resolve()
            )
        else:
            print("Using ensemble as backbone choice: computing RMSD to non-relaxed structure")
            rmsd = compute_rmsd_to_reference_biopython(
                unrelaxed_path.resolve(), relaxed_path.resolve()
            )

        os.remove(unrelaxed_path)

        rmsd_threshold = 1 + np.log(np.sqrt(len("".join(sequence)) / 100))
        if check_clashes == 0 and rmsd < rmsd_threshold:
            print("relaxed structure: no clashes and rmsd under threshold, adding structure")
            return True
        else:
            print(
                f"relaxed structure clashes: {check_clashes} and rmsd: {rmsd}, removing structure"
            )
            os.remove(relaxed_path)
            return False

    # ------------------------------------------------------------------
    # Sidechain sampling
    # ------------------------------------------------------------------

    def sample_sidechains(
        self,
        sequence: list,
        angles: np.ndarray,
        fitted_pops: dict,
        sampler,
    ) -> np.ndarray:
        """Sample chi1 and chi2 angles from the fitted population distributions.

        Parameters
        ----------
        sequence:
            List of one-letter amino-acid codes.
        angles:
            Torsion-angle array to be modified in-place, shape ``(N, 7, 2)``.
        fitted_pops:
            Per-chi population distributions.
        sampler:
            :class:`numpy.random.Generator` instance.

        Returns
        -------
        np.ndarray
            Modified copy of *angles* with sampled chi values.
        """
        sampled_angles = angles.copy()
        sequence = seq2seq("".join(sequence))

        for res in sequence:
            for chi in ["chi1", "chi2"]:
                if self.config.res_has_chi.query(f'restype == "{res[:3]}"')[chi].values[0]:
                    pops = fitted_pops[chi][res]
                    sampled_bin = sampler.choice(len(pops), p=pops)
                    sampled_chi = np.deg2rad(self.config.dihedral_angles[sampled_bin])
                    idx = int(res[3:]) - 1
                    sampled_angles[idx, self.config.dict_chi2layer[chi], 0] = np.sin(sampled_chi)
                    sampled_angles[idx, self.config.dict_chi2layer[chi], 1] = np.cos(sampled_chi)

        return sampled_angles

    # ------------------------------------------------------------------
    # Structure building
    # ------------------------------------------------------------------

    def generate_structure(
        self,
        aatype: jnp.ndarray,
        backbone: r3.Rigids,
        angles: jnp.ndarray,
    ) -> Protein:
        """Build monomer atom coordinates from backbone frames and torsion angles.

        Parameters
        ----------
        aatype:
            Integer amino-acid type array, shape ``(N,)``.
        backbone:
            Monomer backbone rigid-body frames.
        angles:
            Torsion angles, shape ``(N, 7, 2)``.

        Returns
        -------
        Protein
            AlphaFold ``Protein`` object with 37-atom representation.
        """
        all_frames_to_global = torsion_angles_to_frames(aatype, backbone, angles)
        pred_positions = frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global
        )
        atom14_pred_positions = r3.vecs_to_tensor(pred_positions)
        atom37_pred_positions = atom14_to_atom37(
            atom14_pred_positions, slice_batch(self.pdb_input_features, 0)
        )
        return self.from_predicted_atoms(
            self.pdb_input_features,
            np.array(atom37_pred_positions),
            np.array(self.mask_atom),
        )

    def generate_structure_multimer(
        self,
        aatype: jnp.ndarray,
        backbone: geometry.Rigid3Array,
        angles: jnp.ndarray,
    ) -> Protein:
        """Build multimer atom coordinates from backbone frames and torsion angles.

        Parameters
        ----------
        aatype:
            Integer amino-acid type array, shape ``(N,)``.
        backbone:
            Multimer backbone :class:`geometry.Rigid3Array`.
        angles:
            Torsion angles, shape ``(N, 7, 2)``.

        Returns
        -------
        Protein
            AlphaFold ``Protein`` object with 37-atom representation.
        """
        all_frames_to_global = torsion_angles_to_frames_multimer(aatype, backbone, angles)
        atom14_pred_positions = frames_and_literature_positions_to_atom14_pos_multimer(
            aatype, all_frames_to_global
        )
        atom37_pred_positions = atom14_to_atom37_multimer(
            atom14_pred_positions.to_array(), aatype
        )
        return self.from_predicted_atoms(
            self.pdb_input_features,
            np.array(atom37_pred_positions),
            np.array(self.mask_atom),
            remove_leading_feature_dimension=False,
        )

    def from_predicted_atoms(
        self,
        features: FeatureDict,
        final_atom_positions: np.ndarray,
        final_atom_mask: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
        remove_leading_feature_dimension: bool = True,
    ) -> Protein:
        """Assemble a :class:`Protein` from feature arrays and predicted atom positions.

        Adapted from ``alphafold.common.protein``.

        Parameters
        ----------
        features:
            Model input feature dictionary.
        final_atom_positions:
            Predicted all-atom coordinates, shape ``(N, 37, 3)``.
        final_atom_mask:
            Atom presence mask, shape ``(N, 37)``.
        b_factors:
            Optional B-factor array.  Defaults to all zeros.
        remove_leading_feature_dimension:
            When *True* the leading batch dimension is stripped from feature
            arrays (monomer mode); set to *False* for multimer mode.

        Returns
        -------
        Protein
            Fully populated :class:`Protein` instance.
        """
        def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
            return arr[0] if remove_leading_feature_dimension else arr

        if "asym_id" in features:
            chain_index = _maybe_remove_leading_dim(features["asym_id"])
        else:
            chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

        if b_factors is None:
            b_factors = np.zeros_like(final_atom_mask)

        return Protein(
            aatype=_maybe_remove_leading_dim(features["aatype"]),
            atom_positions=final_atom_positions,
            atom_mask=final_atom_mask,
            residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
            chain_index=chain_index,
            b_factors=b_factors,
        )

    def seq2aatype(self, sequence: str) -> list:
        """Convert a one-letter sequence to a list of three-letter residue codes.

        Parameters
        ----------
        sequence:
            One-letter amino-acid sequence.

        Returns
        -------
        list of str
            Three-letter codes as JAX string array.
        """
        return jnp.array(
            [f"{IUPACData.protein_letters_1to3[aa].upper()}" for aa in sequence]
        )


# ---------------------------------------------------------------------------
# RMSD calculation
# ---------------------------------------------------------------------------

def compute_rmsd_to_reference_biopython(reference_pdb, target_pdb) -> float:
    """Compute backbone RMSD between two PDB structures using Biopython.

    The two structures are superimposed on their backbone atoms (N, CA, C)
    before computing the RMSD.

    Parameters
    ----------
    reference_pdb:
        Path to the reference PDB file.
    target_pdb:
        Path to the target PDB file.

    Returns
    -------
    float
        RMSD value in ångströms.
    """
    def extract_backbone_atoms(structure):
        """Extract N, CA, and C atoms from all standard residues."""
        backbone_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":
                        for atom in residue:
                            if atom.id in ["N", "CA", "C"]:
                                backbone_atoms.append(atom)
        return backbone_atoms

    def compute_rmsd(ref_atoms, target_atoms) -> float:
        """Compute RMSD between two lists of Biopython Atom objects."""
        assert len(ref_atoms) == len(target_atoms), "Atom sets must have the same length"
        diff = np.array([a1.coord - a2.coord for a1, a2 in zip(ref_atoms, target_atoms)])
        return np.sqrt((diff ** 2).sum() / len(ref_atoms))

    parser = PDBParser(QUIET=True)

    ref_structure    = parser.get_structure("reference", reference_pdb)
    ref_atoms        = extract_backbone_atoms(ref_structure)

    target_structure = parser.get_structure("target", target_pdb)
    target_atoms     = extract_backbone_atoms(target_structure)

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    super_imposer.apply(target_structure.get_atoms())

    return compute_rmsd(ref_atoms, target_atoms)


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def count_clashes(structure_path, clash_cutoff: float = 0.65) -> int:
    """Count steric clashes in a PDB structure using a KD-tree.

    Clashes within the same residue, across peptide bonds, and in
    disulphide bridges are excluded.

    Parameters
    ----------
    structure_path:
        Path to the PDB file.
    clash_cutoff:
        Fraction of the sum of van-der-Waals radii below which two atoms
        are considered clashing.

    Returns
    -------
    int
        Number of unique clashing atom pairs.
    """
    atom_radii = {
        "C":  1.70,
        "N":  1.55,
        "O":  1.52,
        "S":  1.80,
        "F":  1.47,
        "P":  1.80,
        "CL": 1.75,
        "MG": 1.73,
    }

    clash_cutoffs = {
        f"{i}_{j}": clash_cutoff * (atom_radii[i] + atom_radii[j])
        for i in atom_radii
        for j in atom_radii
    }

    sloppyparser = PDB.PDBParser()
    structure = sloppyparser.get_structure("struct", structure_path)

    atoms  = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    kdt    = PDB.kdtrees.KDTree(coords)

    clashes = []
    for atom_1 in atoms:
        kdt_search = kdt.search(
            np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values())
        )
        potential_clash = [(a.index, a.radius) for a in kdt_search]

        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]

            # Skip intra-residue contacts.
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Skip peptide-bond N–C contacts.
            if (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue
            # Skip disulphide bridges (SG–SG ≤ 2.05 Å is a bond, not a clash).
            if (
                atom_2.name == "SG"
                and atom_1.name == "SG"
                and atom_distance > 1.88
            ):
                continue

            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))

    return len(clashes) // 2


# ---------------------------------------------------------------------------
# Template feature store
# ---------------------------------------------------------------------------

@dataclass
class TemplateStore:
    """Load mmCIF templates and expose them as AlphaFold-compatible feature arrays.

    Builds and holds template features directly from mmCIF files,
    independently of ColabFold's ``get_msa_and_templates`` /
    ``generate_input_feature`` pipeline.

    Usage
    -----
    ::

        store = TemplateStore.build(
            template_dir           = "/path/to/cif/files",
            query_seqs_unique      = query_seqs_unique,
            query_seqs_cardinality = query_seqs_cardinality,
            is_complex             = is_complex,
        )

        # Inspect arrays
        positions = store.atom_positions   # (N, L_total, 37, 3)
        mask      = store.atom_mask        # (N, L_total, 37)
        seq       = store.chain_seq(tmpl=0, chain=0)

        # Use in the model
        store.inject_into(input_features)  # replace pipeline templates
        store.suppress_in(input_features)  # zero-mask pipeline templates

    Parameters
    ----------
    query_seqs_unique:
        List of unique chain sequences (one-letter codes).
    query_seqs_cardinality:
        How many copies of each unique sequence appear in the complex.
    is_complex:
        Whether the target is a multimeric complex.
    """

    # --- inputs ---
    query_seqs_unique:      list
    query_seqs_cardinality: list
    is_complex:             bool

    # --- derived, populated by __post_init__ and build() ---
    chains:            list      = field(init=False, default_factory=list)
    chain_lengths:     list      = field(init=False, default_factory=list)
    offsets:           np.ndarray = field(init=False, default=None)
    L_total:           int       = field(init=False, default=0)
    features:          dict      = field(init=False, default_factory=dict)
    per_template_info: list      = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __post_init__(self):
        """Pre-compute the concatenated chain layout from unique seqs + cardinality."""
        self.chains = []
        for seq, card in zip(self.query_seqs_unique, self.query_seqs_cardinality):
            self.chains.extend([seq] * card)
        self.chain_lengths = [len(s) for s in self.chains]
        self.L_total       = sum(self.chain_lengths)
        self.offsets       = np.cumsum([0] + self.chain_lengths[:-1]).astype(int)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        template_dir:           "str | Path",
        query_seqs_unique:      list,
        query_seqs_cardinality: list,
        is_complex:             bool = False,
    ) -> "TemplateStore":
        """Construct a :class:`TemplateStore` by parsing all ``.cif`` files in a directory.

        Parameters
        ----------
        template_dir:
            Directory containing one or more mmCIF (``.cif``) template files.
        query_seqs_unique:
            List of unique chain sequences.
        query_seqs_cardinality:
            Copies of each unique chain in the complex.
        is_complex:
            Enable multi-chain template matching when *True*.

        Returns
        -------
        TemplateStore
            Fully populated store with ``features`` ready for injection.

        Raises
        ------
        ValueError
            If no ``.cif`` files are found in *template_dir*.
        """
        store     = cls(query_seqs_unique, query_seqs_cardinality, is_complex)
        cif_files = sorted(Path(template_dir).glob("*.cif"))
        N         = len(cif_files)
        n_chains  = len(store.chains)

        if N == 0:
            raise ValueError(f"No .cif files found in {template_dir}")

        all_atom_positions = np.zeros((N, store.L_total, 37, 3), dtype=np.float32)
        all_atom_mask      = np.zeros((N, store.L_total, 37),    dtype=np.float32)
        template_aatype    = np.full( (N, store.L_total), 20,    dtype=np.int32)
        domain_names       = []

        parser = MMCIFParser(QUIET=True)

        for i, cif_file in enumerate(cif_files):
            info = {
                "name":       cif_file.stem,
                "file":       cif_file,
                "chain_seqs": [""] * n_chains,
            }
            try:
                structure    = parser.get_structure("t", str(cif_file))
                model        = next(structure.get_models())
                mmcif_chains = list(model.get_chains())

                if not is_complex:
                    # Monomer: first chain only, always placed at offset 0.
                    seq = store._fill_chain(
                        chain          = mmcif_chains[0],
                        atom_positions = all_atom_positions[i],
                        atom_mask      = all_atom_mask[i],
                        aatype         = template_aatype[i],
                        offset         = 0,
                        length         = store.chain_lengths[0],
                    )
                    info["chain_seqs"][0] = seq

                else:
                    # Multimer: match each mmCIF chain to the best query slot
                    # by sequence identity.
                    used_query_slots = set()

                    for mmcif_chain in mmcif_chains:
                        tmpl_seq = cls._extract_sequence(mmcif_chain)
                        if not tmpl_seq:
                            continue

                        best_slot, best_score = cls._best_matching_chain(
                            tmpl_seq, store.chains, used_query_slots
                        )

                        if best_slot is None:
                            print(
                                f"[TemplateStore] {cif_file.name}: template chain "
                                f"{mmcif_chain.id!r} (len {len(tmpl_seq)}) "
                                f"could not be matched to any query chain — skipping"
                            )
                            continue

                        used_query_slots.add(best_slot)
                        offset = int(store.offsets[best_slot])
                        length = store.chain_lengths[best_slot]

                        seq = store._fill_chain(
                            chain          = mmcif_chain,
                            atom_positions = all_atom_positions[i],
                            atom_mask      = all_atom_mask[i],
                            aatype         = template_aatype[i],
                            offset         = offset,
                            length         = length,
                        )
                        info["chain_seqs"][best_slot] = seq

            except Exception as e:
                print(f"[TemplateStore] Warning: skipping {cif_file.name}: {e}")

            domain_names.append(cif_file.stem.encode())
            store.per_template_info.append(info)

        store.features = {
            "template_all_atom_positions": all_atom_positions,
            "template_all_atom_mask":      all_atom_mask,
            "template_aatype":             template_aatype,
            "template_domain_names":       np.array(domain_names),
            "template_sum_probs":          np.zeros((N, 1), dtype=np.float32),
        }
        return store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sequence(chain) -> str:
        """Extract the one-letter sequence from a Biopython chain.

        Parameters
        ----------
        chain:
            A :class:`Bio.PDB.Chain.Chain` object.

        Returns
        -------
        str
            One-letter amino-acid sequence; unknown residues map to ``"X"``.
        """
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        seq = [
            residue_constants.restype_3to1.get(r.get_resname().strip(), "X")
            for r in residues
        ]
        return "".join(seq)

    @staticmethod
    def _best_matching_chain(
        tmpl_seq:     str,
        query_chains: list,
        exclude:      set,
        min_identity: float = 0.30,
    ) -> tuple:
        """Match a template chain sequence to the closest available query chain slot.

        Parameters
        ----------
        tmpl_seq:
            One-letter sequence of the template chain.
        query_chains:
            List of unique query-chain sequences.
        exclude:
            Set of slot indices already assigned.
        min_identity:
            Minimum sequence identity required for a valid match.

        Returns
        -------
        tuple[int | None, float]
            ``(best_slot, identity)`` — ``best_slot`` is *None* if no chain
            meets the *min_identity* threshold.
        """
        best_slot   = None
        best_score  = 0.0
        best_lscore = 0.0

        for slot_idx, query_seq in enumerate(query_chains):
            if slot_idx in exclude:
                continue

            len_t = len(tmpl_seq)
            len_q = len(query_seq)
            if len_t == 0 or len_q == 0:
                continue

            min_len  = min(len_t, len_q)
            max_len  = max(len_t, len_q)
            matches  = sum(t == q for t, q in zip(tmpl_seq[:min_len], query_seq[:min_len]))
            identity = matches / max_len
            lscore   = 1.0 - abs(len_t - len_q) / max_len

            if identity > best_score or (
                identity == best_score and lscore > best_lscore
            ):
                best_score  = identity
                best_lscore = lscore
                best_slot   = slot_idx

        if best_score < min_identity:
            return None, 0.0

        return best_slot, best_score

    @staticmethod
    def _fill_chain(
        chain,
        atom_positions: np.ndarray,
        atom_mask:      np.ndarray,
        aatype:         np.ndarray,
        offset:         int,
        length:         int,
    ) -> str:
        """Populate per-residue arrays in-place for one chain.

        Parameters
        ----------
        chain:
            Biopython chain to read from.
        atom_positions:
            All-atom coordinate array to fill, shape ``(L_total, 37, 3)``.
        atom_mask:
            Atom presence mask to fill, shape ``(L_total, 37)``.
        aatype:
            Integer amino-acid type array to fill, shape ``(L_total,)``.
        offset:
            Residue index in the concatenated sequence where this chain starts.
        length:
            Maximum number of residues to process from this chain.

        Returns
        -------
        str
            One-letter sequence extracted from the mmCIF chain.
        """
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        seq = []

        for local_idx, residue in enumerate(residues):
            if local_idx >= length:
                break

            global_idx = offset + local_idx
            resname    = residue.get_resname().strip()
            one_letter = residue_constants.restype_3to1.get(resname, "X")
            seq.append(one_letter)
            aatype[global_idx] = residue_constants.restype_order.get(one_letter, 20)

            for atom in residue.get_atoms():
                atom_name = atom.get_name().strip()
                if atom_name in residue_constants.atom_order:
                    atom_idx = residue_constants.atom_order[atom_name]
                    atom_positions[global_idx, atom_idx] = atom.get_vector().get_array()
                    atom_mask[global_idx, atom_idx] = 1.0

        return "".join(seq)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def atom_positions(self) -> np.ndarray:
        """All-atom coordinates for every template, shape ``(N, L_total, 37, 3)``."""
        return self.features["template_all_atom_positions"]

    @property
    def atom_mask(self) -> np.ndarray:
        """Atom presence mask for every template, shape ``(N, L_total, 37)``."""
        return self.features["template_all_atom_mask"]

    @property
    def n_templates(self) -> int:
        """Number of templates currently stored."""
        return len(self.per_template_info)

    def chain_seq(self, tmpl: int, chain: int) -> str:
        """Return the one-letter sequence for *chain* of template *tmpl*.

        Parameters
        ----------
        tmpl:
            Template index (0-based).
        chain:
            Chain slot index (0-based).
        """
        return self.per_template_info[tmpl]["chain_seqs"][chain]

    def template_name(self, tmpl: int) -> str:
        """Return the stem of the source mmCIF file for template *tmpl*."""
        return self.per_template_info[tmpl]["name"]

    def slice_chain(self, tmpl: int, chain: int) -> tuple:
        """Return atom positions and mask sliced to the residue range of one chain.

        Parameters
        ----------
        tmpl:
            Template index (0-based).
        chain:
            Chain slot index (0-based).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(positions, mask)`` with shapes ``(L_chain, 37, 3)`` and
            ``(L_chain, 37)`` respectively.
        """
        start = int(self.offsets[chain])
        end   = start + self.chain_lengths[chain]
        return (
            self.atom_positions[tmpl, start:end],
            self.atom_mask[tmpl, start:end],
        )

    # ------------------------------------------------------------------
    # Integration with ColabFold input features
    # ------------------------------------------------------------------

    def inject_into(self, input_features: dict) -> dict:
        """Replace template keys in *input_features* with this store's arrays.

        Call this to make the model use your custom templates instead of
        whatever ``generate_input_feature`` produced.  Modifies
        *input_features* in-place and returns it.

        Parameters
        ----------
        input_features:
            ColabFold/AlphaFold input-feature dictionary.

        Returns
        -------
        dict
            The same dictionary with template features replaced.
        """
        for key in [
            "template_all_atom_positions",
            "template_all_atom_mask",
            "template_aatype",
            "template_domain_names",
            "template_sum_probs",
        ]:
            if key in self.features:
                input_features[key] = self.features[key]
        return input_features

    def suppress_in(self, input_features: dict) -> dict:
        """Zero out template keys so the model ignores pipeline templates.

        The store still holds the real data for inspection.  Modifies
        *input_features* in-place and returns it.

        Parameters
        ----------
        input_features:
            ColabFold/AlphaFold input-feature dictionary.

        Returns
        -------
        dict
            The same dictionary with template features zeroed.
        """
        input_features["template_all_atom_mask"][:] = 0
        input_features["template_all_atom_positions"][:] = 0
        if "template_pseudo_beta_mask" in input_features:
            input_features["template_pseudo_beta_mask"][:] = 0
        if "template_aatype" in input_features:
            input_features["template_aatype"][:] = 20
        return input_features

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_templates

    def __repr__(self) -> str:
        names = [info["name"] for info in self.per_template_info]
        return (
            f"TemplateStore("
            f"n_templates={self.n_templates}, "
            f"L_total={self.L_total}, "
            f"chains={len(self.chains)}, "
            f"templates={names})"
        )

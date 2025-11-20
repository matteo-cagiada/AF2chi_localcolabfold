import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from scipy.stats import circmean, circstd
from scipy.optimize import minimize

from Bio.Data import IUPACData
from Bio import PDB
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Atom import Atom

import multiprocessing as mp
from multiprocessing import Pool

import warnings
from typing import Any, Mapping, Optional, Dict

import ml_collections
from ml_collections import ConfigDict

from alphafold.common.protein import Protein, to_pdb, from_pdb_string
from alphafold.model.all_atom import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos,
    atom14_to_atom37,
)
from alphafold.model.all_atom_multimer import (
    torsion_angles_to_frames as torsion_angles_to_frames_multimer,
    frames_and_literature_positions_to_atom14_pos as frames_and_literature_positions_to_atom14_pos_multimer,
    atom14_to_atom37 as atom14_to_atom37_multimer,
)

from alphafold.model import quat_affine, r3, geometry
from alphafold.model.geometry.rigid_matrix_vector import Rigid3Array

from alphafold.common import residue_constants
import colabfold.relax_sc as relax
import warnings
from colabfold.download import default_data_dir

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called")

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.


def get_config(
    path_rot_lib: str = None,
    path_rot_sigmas: str = None,
    path_chi_checks: str = None,
    struct_weight: float = 0.85,
    n_struct_ensemble=100,
    use_gpu_relax: bool = True
) -> ConfigDict:
    print(">>>>>>> structure_weight:", struct_weight)
    config = ConfigDict()

    config.dihedral_angles = [10 * i for i in range(36)]
    config.layers = [1, 2, 3, 4, 5, 6]

    config.struct_weight = struct_weight
    config.thetas = np.geomspace(1, 10000, num=20)
    config.dihedral_angles = [5 + (10 * i) for i in range(36)]
    config.n_struct_ensemble = int(n_struct_ensemble)
    #config.pool_cpus = int(os.cpu_count() / 4)
    config.pool_cpus = max(os.cpu_count(), int(os.cpu_count() / 4))

    #set default af2chi param dir
    default_af2chi_param_dir = os.path.join(default_data_dir, "af2chi-params")
    
    config.use_gpu_relax = use_gpu_relax

    if path_rot_lib != None:
        config.rot_lib = pd.read_csv(path_rot_lib, index_col=0)
    else:
        config.rot_lib = pd.read_csv(
                os.path.join(default_af2chi_param_dir,"Top8000_rebinned_all_chi_distributions.csv"),
            index_col=0,
        )

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
    }  # chi1 is the 4th dihedral angle in the internal layer, the higher chis are the higher dimensions layers, layers are 0-indexed

    return config

def class_to_np(c):
    """Recursively changes jax arrays to numpy arrays."""

    class dict2obj:
        def __init__(self, d):
            for k, v in _jnp_to_np(d).items():
                setattr(self, k, v)

    return dict2obj(c.__dict__)


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def slice_batch(batch, i):
    """Slice a batch of tensors."""
    return {k: v[i] for k, v in batch.items()}


def seq2seq(sequence: str) -> list:
    """convert a 1-letter sequence to a 3-letter list"""
    return [
        f"{IUPACData.protein_letters_1to3[aa].upper()}{i + 1}"
        for i, aa in enumerate(sequence)
    ]


def patch_openmm():
    from openmm import app
    from openmm.unit import nanometers, sqrt

    # applied https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch
    # to OpenMM 7.7.1 (see PR https://github.com/openmm/openmm/pull/3203)
    # patch is licensed under CC-0
    # OpenMM is licensed under MIT and LGPL
    # fmt: off
    def createDisulfideBonds(self, positions):
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return 'SG' in names and 'HG' not in names
        # This function is used to prevent multiple di-sulfide bonds from being
        # assigned to a given atom.
        def isDisulfideBonded(atom):
            for b in self._bonds:
                if (atom in b and b[0].name == 'SG' and
                    b[1].name == 'SG'):
                    return True

            return False

        cyx = [res for res in self.residues() if res.name == 'CYS' and isCyx(res)]
        atomNames = [[atom.name for atom in res._atoms] for res in cyx]
        for i in range(len(cyx)):
            sg1 = cyx[i]._atoms[atomNames[i].index('SG')]
            pos1 = positions[sg1.index]
    # applied https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch
    # to OpenMM 7.7.1 (see PR https://github.com/openmm/openmm/pull/3203)
    # patch is licensed under CC-0
    # OpenMM is licensed under MIT and LGPL
    # fmt: off
    def createDisulfideBonds(self, positions):
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return 'SG' in names and 'HG' not in names
        # This function is used to prevent multiple di-sulfide bonds from being
        # assigned to a given atom.
        def isDisulfideBonded(atom):
            for b in self._bonds:
                if (atom in b and b[0].name == 'SG' and
                    b[1].name == 'SG'):
                    return True

            return False

        cyx = [res for res in self.residues() if res.name == 'CYS' and isCyx(res)]
        atomNames = [[atom.name for atom in res._atoms] for res in cyx]
        for i in range(len(cyx)):
            sg1 = cyx[i]._atoms[atomNames[i].index('SG')]
            pos1 = positions[sg1.index]
            candidate_distance, candidate_atom = 0.3*nanometers, None
            for j in range(i):
                sg2 = cyx[j]._atoms[atomNames[j].index('SG')]
                pos2 = positions[sg2.index]
                delta = [x-y for (x,y) in zip(pos1, pos2)]
                distance = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2])
                if distance < candidate_distance and not isDisulfideBonded(sg2):
                    candidate_distance = distance
                    candidate_atom = sg2
            # Assign bond to closest pair.
            if candidate_atom:
                self.addBond(sg1, candidate_atom)
    # fmt: on
    app.Topology.createDisulfideBonds = createDisulfideBonds


def relax_sidechains(
    pdb_filename=None,
    sampled_angles=None,
    config=None,
    pdb_lines=None,
    pdb_obj=None,
    use_gpu=False,
    max_iterations=0,
    stiffness=1.0,
):  # sc start #Emil added sampled_angles pass
    if "relax" not in dir():  # sc
        patch_openmm()
    if pdb_obj is None:
        if pdb_lines is None:
            pdb_lines = Path(pdb_filename).read_text()
        pdb_obj = from_pdb_string(pdb_lines)

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=max_iterations,
        tolerance=2.39,
        stiffness=stiffness,  # 10.0 originally
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu,
        sampled_angles=sampled_angles,  # Emil
        config=config,
    )  # Emil added sampled_angles pass

    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)
    return relaxed_pdb_lines  # sc end


class af2sidechain_pops:
    """Class to calculate the chi populations of a protein sequence given the predicted angles and the sequence"""

    def __init__(self, config, is_complex=False):
        self.config = config
        self.is_complex = is_complex

    def __call__(self, sequence: list, angles: np.array, chi_sel: str):
        """Args:
        sequence: list of str, protein sequence
        angles: np.array, predicted angles
        chi_sel: str, selected chi to calculate the populations
        """
        
        # extract chi1 internal layer median and final values
        #print(
        #    f" chi_sel: {chi_sel}, meaning layer: {self.config.dict_chi2layer[chi_sel]}" ## uncomment for debugging
        #)

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

        return (fitted_pops, prior_pops)

    def af2chi(
        self, angles: np.array, layer_chi: int
    ) -> (np.array, np.array, np.array):
        """Convert AF2 sidechains prediction layers to chi median + std (internal layers) and last layer values
        Args:
        angles: np.array, predicted angles from AF2 (batch, layers, 37, 2)
        layer_chi: int, which chi to extract
        """
        ### chi1 is the 4th dihedral angle in the internal layer, the higher chis are the higher dimensions layers

        chi_internal_layers = self.extract_chi_angles(
            angles, layer_chi, self.config.layers, False
        ).astype(np.float64)

        ## calculate mean and std of chi1 internal layers
        chi_internal_layers_mean = circmean(
            chi_internal_layers, low=0, high=2 * np.pi, axis=0
        )

        chi_internal_layers_std = circstd(
            chi_internal_layers, low=0, high=2 * np.pi, axis=0
        )

        ## extract chi1 final layer (i.e. structure chi1s)
        chi_final_layer = self.extract_chi_angles(
            angles, layer_chi, is_last=True
        ).flatten()

        return chi_internal_layers_mean, chi_internal_layers_std, chi_final_layer

    def extract_chi_angles(
        self, angles: np.array, layer_chi: int, layers: list = [], is_last: bool = False
    ) -> np.array:
        """Extract chi angles from the predicted angles
        Args:
        angles: np.array, predicted angles from AF2 (batch, layers, 37, 2)
        layer_chi: int, which chi to extract
        layers: list, which layers to extract chi angles from
        is_last: bool, if True, only extract the last layer
        """

        chi_angles = []

        ## if last layer, only extract the last layer
        if is_last:
            layers = [-1]
        ## extract chi1 angles from each layer
        for layer in layers:
            chi_angles.append(
                circmean(
                    np.arctan2(
                        angles[layer][:, layer_chi, 0], angles[layer][:, layer_chi, 1]
                    ).reshape(-1, 1),
                    low=0,
                    high=2 * np.pi,
                    axis=1,
                )
            )

        return np.array(chi_angles)

    def eval_chi_populations(
        self,
        sequence: list,
        chi_sel,
        chi_internal_layers_mean: np.array,
        chi_internal_layers_std: np.array,
        chi_final_layer: np.array,
    ) -> dict:
        """Calculate the chi populations of the protein sequence given the predicted information extracted with af2chi
        Args:
        sequence: list of str, protein sequence
        chi_sel: str, selected chi to calculate the populations
        chi_internal_layers_mean: np.array, median of the internal layers
        chi_internal_layers_std: np.array, std of the internal layers
        chi_final_layer: np.array, final layer values
        """

        res_list_with_sel_chi = self.res_list_with_sel_chi(
            sequence, chi_sel
        )  ## join used to deal with multi-chain proteins

        prior_comb = self.create_combined_prior(
            res_list_with_sel_chi,
            chi_sel,
            chi_final_layer,
            sigmas_restype=self.config.sigmas_restype,
            structure_weight=self.config.struct_weight,
            rot_lib=self.config.rot_lib,
            dihedral_angles=self.config.dihedral_angles,
        )

        fitted_pops_residues, prior_pops_residues = {}, {}
        #mp.set_start_method("spawn", force=True)

        pool = Pool(self.config.pool_cpus)

        results = pool.starmap(
            self.maxent_rw_populations,
            [
                (
                    res,
                    chi_internal_layers_mean,
                    chi_internal_layers_std,
                    prior_comb,
                    self.config,
                )
                for res in res_list_with_sel_chi
            ],
        )

        pool.close()
        pool.join()

        for res, fitted_pops, prior_pops in results:
            fitted_pops_residues[res] = list(fitted_pops)
            prior_pops_residues[res] = list(prior_pops)

        return fitted_pops_residues, prior_pops_residues

    def create_combined_prior(
        self,
        sequence: list,
        chi_sel: str,
        chi_final_layer: np.array,
        sigmas_restype: dict,
        structure_weight: float,
        rot_lib: dict,
        dihedral_angles: list,
    ) -> dict:
        """Create a combined prior as the input for the reweighting from literature priori and last layer chi values
        Args:
        sequence: list of str, protein sequence
        chi_sel: str, selected chi to calculate the populations
        chi_final_layer: np.array, final layer values
        sigmas_restype: dict, sigmas of the chi distributions
        structure_weight: float, weight of the structure prior
        rot_lib: dict, rotamer library
        dihedral_angles: list, dihedral angles
        """

        def gaussian(x, a, mu, sigma):
            y = (
                a
                * 1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * (np.square(x - mu) / np.square(sigma)))
            )
            return y

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
            # try:
            w0_structure = gaussian(
                dihedral_angles,
                1,
                chi_final_layer_deg,
                sigmas_restype[restype + "_" + chi_sel],
            )
            w0_structure /= np.sum(w0_structure)
            w0_rot_lib = np.array(rot_lib[restype + "_" + chi_sel])
            w0_rot_lib /= np.sum(w0_rot_lib)
            w0_comb = (
                structure_weight * w0_structure + (1 - structure_weight) * w0_rot_lib
            )
            w0_comb /= np.sum(w0_comb)
            # except KeyError:
            #    w0_comb = np.empty(len(dihedral_angles), dtype=float)
            #    w0_comb[:] = np.nan

            prior_combined[residue] = w0_comb

        return prior_combined

    def res_list_with_sel_chi(self, sequence: list, chi_sel: str) -> list:
        """Get the residues which have the selected chi
        Args:
        sequence: list of str, protein sequence
        chi_sel: str, selected chi to calculate the populations
        """

        return [
            res
            for res in sequence
            if self.config.res_has_chi.query(f'restype == "{res[:3]}"')[chi_sel].values[
                0
            ]
            == True
        ]

    def maxent_rw_populations(
        self,
        res: str,
        chi_internal_layers_mean: np.array,
        chi_internal_layers_std: np.array,
        prior_combined: dict,
        config: ml_collections.ConfigDict,
    ) -> [list, np.array, np.array]:
        """Calculate the chi populations of a residue using the maxent reweighting method starting from a prior
        Args:
        res: str, residue
        chi_internal_layers_mean: np.array, median of the internal layers
        chi_internal_layers_std: np.array, std of the internal layers
        prior_combined: dict, combined prior
        config: ml_collections.ConfigDict, configuration
        """

        warnings.filterwarnings("ignore")

        def maxent_loss(
            w1, w0, dihedral_angles, dihedral_avg_AF, dihedral_avg_AF_err, theta
        ):
            """Maxent loss function"""
            dihedral_avg_calc = np.rad2deg(
                circweightedmean(np.deg2rad(dihedral_angles), weights=w1)
            )
            chisquare = eval_chisquare(
                dihedral_avg_calc, dihedral_avg_AF, dihedral_avg_AF_err
            )
            L = 0.5 * chisquare - theta * (-np.sum(w1 * np.log(w1 / w0)))

            return L / (theta)

        def eval_chisquare(angle1: float, angle2: float, angle_err: float) -> float:
            """Calculate chi2"""
            return np.square(
                diff_angles(np.deg2rad(angle1), np.deg2rad(angle2))
                / np.deg2rad(angle_err)
            )

        def theta_loc(thetas, chisquare):
            """Select theta"""
            # Choose highest theta that gives chi2 <= 1.0
            if np.any(chisquare <= 1.0):
                idx = np.argwhere(chisquare <= 1.0).flatten()[-1]

            # if there is not chi2 <= 1.0, choose the theta that gives lowest chi2
            else:
                idx = np.argmin(chisquare)

            # Get selected theta, phieff and chi2 and return
            return thetas[idx], idx

        def diff_angles(angle1: float, angle2: float) -> float:
            """Calculate the difference between two angles"""
            return np.angle(np.exp(1j * (angle1 - angle2)))

        def circweightedmean(angles, weights=None):
            """Calculate the circular weighted mean of a set of angles"""
            if weights is None:
                weights = np.array([1] * len(angles))

            weights_norm = weights / np.sum(weights)

            if np.all(weights_norm == weights_norm[0]):
                sin_avg = 0
                cos_avg = 0
            else:
                sin_avg = np.sum(weights_norm * np.sin(angles))
                cos_avg = np.sum(weights_norm * np.cos(angles))

            angle_avg = np.arctan2(sin_avg, cos_avg)

            return angle_avg

        resnum = int(res[3:])
        restype = res[:3]

        chi_avg_target = np.rad2deg(chi_internal_layers_mean[resnum - 1])
        chi_err_target = 2 * float(np.rad2deg(chi_internal_layers_std[resnum - 1]))

        if np.any(prior_combined[res] == np.nan):

            return [res, prior_combined[res], prior_combined[res]]

        else:

            w0 = prior_combined[res]
            ### weights cannot be zero in maxent
            w0[w0 == 0.0] = 1e-32

            guess = w0

            bnds = ((1e-32, 1),) * len(w0)

            # weights should sum to 1
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

            fitted_pops_vs_theta = []
            chisquare_vs_theta = np.zeros(len(config.thetas))
            phieff_vs_theta = np.zeros(len(config.thetas))

            for j, theta in enumerate(config.thetas):
                # Set args
                args = (
                    w0,
                    config.dihedral_angles,
                    chi_avg_target,
                    chi_err_target,
                    theta,
                )  # , alpha)

                # Minimize Maxent loss with bounds and constraint defined above
                min_result = minimize(
                    maxent_loss,
                    guess,
                    args=args,
                    bounds=bnds,
                    constraints=constraints,
                    method="SLSQP",
                    options={"ftol": 1e-12},
                )

                # Get fitted weights from minimization and renormalize just to be safe
                weights_fit = min_result["x"]
                weights_fit /= np.sum(weights_fit)

                # Get average dihedral from fitted populations

                # Get chi2
                chisquare_vs_theta[j] = eval_chisquare(
                    np.rad2deg(
                        circweightedmean(
                            np.deg2rad(config.dihedral_angles), weights=weights_fit
                        )
                    ),
                    chi_avg_target,
                    chi_err_target,
                )

                # Get phieff
                fitted_pops_vs_theta.append(weights_fit)
            theta_sel, idx = theta_loc(config.thetas, chisquare_vs_theta)
            return [res, np.array(fitted_pops_vs_theta[idx]), np.array(w0)]


class create_pdb_ensemble:
    """Class to generate a PDB ensemble from given backbone and predicted chi populations"""

    def __init__(
        self,
        config,
        pdb_input_features,
        mask_atom,
        result_dir,
        ref_pdb_path,
        is_complex=False,
        stiffness=1.0,
    ):
        self.config = config
        self.pdb_input_features = pdb_input_features
        self.mask_atom = mask_atom
        self.result_dir = result_dir.joinpath("sidechain_ensemble")
        self.sampler = np.random.default_rng()
        self.reference_pdb_path = ref_pdb_path
        self.is_complex = is_complex
        self.stiffness = stiffness  ## temporary, we need to decide what to do with it

    def __call__(
        self,
        sequence: list,
        backbone: jnp.array,
        angles: jnp.array,
        fitted_pops: dict,
        jobname: str,
    ) -> np.array:
        """Args:
        sequence: list of str, protein sequence
        backbone: jnp.array, predicted backbone
        angles: np.array, predicted angles
        fitted_pops: dict, fitted chi populations
        jobname: str, jobname
        """

        ## while loop while generating ensemble, with one structure at the time create, checked for clashes, added to ensemble and check ensemble requirements

        if self.is_complex:
            r3_bb = Rigid3Array.from_array(backbone.astype(jnp.float32))
            aatype = jnp.array(self.pdb_input_features["aatype"])
        else:
            ### transform the saved bakbone into Quaternion and then into Rigid (see classes in AF2 for explanation) --> need to run structure generation
            quat_bb = quat_affine.QuatAffine.from_tensor(jnp.array(backbone))
            r3_bb = r3.rigids_from_quataffine(quat_bb)
            aatype = jnp.array(self.pdb_input_features["aatype"][0])

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        angles = angles[-1, ...]
        ensemble = 0

        # while it  <  self.config.max_iter:
        production_pool = []

        print(f"Generating ensemble of {self.config.n_struct_ensemble} structures")
        max_size_try = 3 * self.config.n_struct_ensemble
        # while ensemble < (it+1)*100: ##self.config.max_iter:
        while ensemble < self.config.n_struct_ensemble:

            ## sample sidechains
            print(f">>> Generating ensemble structure {ensemble} ...")
            production_pool.append(
                self.ensemble_creation_iteration(
                    sequence, aatype, r3_bb, angles, fitted_pops, jobname, ensemble
                )
            )
            if production_pool[-1]:
                ensemble += 1
            if len(production_pool) > max_size_try:
                print(f"max try for ensemble generation reached, breaking")
                break

        return production_pool

    def ensemble_creation_iteration(
        self,
        sequence: list,
        aatype: np.array,
        r3_bb: jnp.array,
        angles: jnp.array,
        fitted_pops: dict,
        jobname: str,
        ensemble: int
    ) -> bool:
        """Args:
        sequence: list of str, protein sequence
        aatype: np.array, amino acid types
        r3_bb: jnp.array, backbone
        angles: np.array, predicted angles
        fitted_pops: dict, fitted chi populations
        jobname: str, jobname
        ensemble: int, ensemble number
        """

        ## sample sidechains
        sampled_angles = self.sample_sidechains(
            sequence, angles, fitted_pops, self.sampler
        )

        ## generate structure
        jnp_sampled_angles = jnp.array(sampled_angles)
        if self.is_complex:
            structure = self.generate_structure_multimer(
                aatype, r3_bb, jnp_sampled_angles
            )
        else:
            structure = self.generate_structure(aatype, r3_bb, jnp_sampled_angles)

        pdb_lines = to_pdb(class_to_np(structure))
        Path(
            self.result_dir.joinpath(jobname + f"_sidechain_ensemble_{ensemble}.pdb")
        ).write_text(pdb_lines)

        check_clashes = count_clashes(
            self.result_dir.joinpath(jobname + f"_sidechain_ensemble_{ensemble}.pdb")
        )
        print(f"non-relaxed structure clashes: {check_clashes}")

        ## relax sidechains

        relaxed_pdb_lines = relax_sidechains(
            pdb_lines=pdb_lines,
            sampled_angles=sampled_angles,
            config=self.config,
            use_gpu=self.config.use_gpu_relax,
            max_iterations=0,
            stiffness=self.stiffness,
        )  # Emil added sampled_angles pass ## stiffness temporary we need to decide what to do with it
        Path(
            self.result_dir.joinpath(
                jobname + f"_sidechain_ensemble_relaxed_{ensemble}.pdb"
            )
        ).write_text(relaxed_pdb_lines)

        os.remove(
            self.result_dir.joinpath(jobname + f"_sidechain_ensemble_{ensemble}.pdb")
        )

        check_clashes = count_clashes(
            self.result_dir.joinpath(
                jobname + f"_sidechain_ensemble_relaxed_{ensemble}.pdb"
            )
        )
        rmsd = compute_rmsd_to_reference_biopython(
            self.reference_pdb_path,
            self.result_dir.joinpath(
                jobname + f"_sidechain_ensemble_relaxed_{ensemble}.pdb"
            ).resolve(),
        )

        if check_clashes == 0 and rmsd < (
            1 + np.log(np.sqrt(len("".join(sequence)) / 100))
        ):
            print(
                f"relaxed structure: no clashes and rmsd under threshold, adding structure"
            )
            return True
        else:
            print(
                f"relaxed structure clashes: {check_clashes} and rsmd: {rmsd}, removing structure"
            )
            os.remove(
                self.result_dir.joinpath(
                    jobname + f"_sidechain_ensemble_relaxed_{ensemble}.pdb"
                )
            )
            return False

    def sample_sidechains(
        self, sequence: list, angles: np.array, fitted_pops: dict, sampler
    ) -> np.array:

        sampled_angles = angles.copy()

        sequence = seq2seq("".join(sequence))

        for res in sequence:
            for chi in ["chi1", "chi2"]:
                if (
                    self.config.res_has_chi.query(f'restype == "{res[:3]}"')[
                        chi
                    ].values[0]
                    == True
                ):

                    pops = fitted_pops[chi][res]
                    sampled_bin = sampler.choice(len(pops), p=pops)
                    sampled_chi = np.deg2rad(self.config.dihedral_angles[sampled_bin])
                    ### update sampled angles (sin, cos)
                    sampled_angles[
                        int(res[3:]) - 1, self.config.dict_chi2layer[chi], 0
                    ] = np.sin(sampled_chi)
                    sampled_angles[
                        int(res[3:]) - 1, self.config.dict_chi2layer[chi], 1
                    ] = np.cos(sampled_chi)

        return sampled_angles

    def generate_structure(
        self, aatype: jnp.array, backbone: r3.Rigids, angles: jnp.array
    ) -> np.array:

        all_frames_to_global = torsion_angles_to_frames(aatype, backbone, angles)

        # Use frames and literature positions to create the final atom coordinates.
        # geometry.Vec3Array with shape (N, 14)
        pred_positions = frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global
        )

        atom14_pred_positions = r3.vecs_to_tensor(pred_positions)
        atom37_pred_positions = atom14_to_atom37(
            atom14_pred_positions, slice_batch(self.pdb_input_features, 0)
        )
        structure = self.from_predicted_atoms(
            self.pdb_input_features,
            np.array(atom37_pred_positions),
            np.array(self.mask_atom),
        )

        ## generate structure

        return structure

    def generate_structure_multimer(
        self, aatype: jnp.array, backbone: geometry.Rigid3Array, angles: jnp.array
    ) -> np.array:

        all_frames_to_global = torsion_angles_to_frames_multimer(
            aatype, backbone, angles
        )

        # Use frames and literature positions to create the final atom coordinates.
        # geometry.Vec3Array with shape (N, 14)
        atom14_pred_positions = frames_and_literature_positions_to_atom14_pos_multimer(
            aatype, all_frames_to_global
        )
        atom37_pred_positions = atom14_to_atom37_multimer(
            atom14_pred_positions.to_array(), aatype
        )
        structure = self.from_predicted_atoms(
            self.pdb_input_features,
            np.array(atom37_pred_positions),
            np.array(self.mask_atom),
            remove_leading_feature_dimension=False,
        )

        ## generate structure
        return structure

    def from_predicted_atoms(
        self,
        features: FeatureDict,
        final_atom_positions: np.ndarray,
        final_atom_mask: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
        remove_leading_feature_dimension: bool = True,
    ) -> Protein:
        """Assembles a protein from a prediction. (adapted from alphafold.common.protein)

        Args:
        features: Dictionary holding model inputs.
        result: Dictionary holding model outputs.
        b_factors: (Optional) B-factors to use for the protein.
        remove_leading_feature_dimension: Whether to remove the leading dimension
            of the `features` values.

        Returns:
        A protein instance.
        """ ""

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
        return jnp.array(
            [f"{IUPACData.protein_letters_1to3[aa].upper()}" for aa in sequence]
        )

    def generate_sidechains(self, fitted_pops: dict) -> jnp.array:

        ## generate sidechains

        return structure


def compute_rmsd_to_reference_biopython(reference_pdb, target_pdb):
    """
    Compute the RMSD between two PDB structures using Biopython
    Args:
    reference_pdb: str, path to the reference PDB file
    target_pdb: str, path to the target PDB file
    Returns:
    float: RMSD value
    """

    def extract_backbone_atoms(structure):
        """Extract backbone atoms (N, CA, C) from a Biopython structure."""
        backbone_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if it's an amino acid
                    if residue.id[0] == " ":
                        for atom in residue:
                            if atom.id in [
                                "N",
                                "CA",
                                "C",
                            ]:  # Select backbone atoms (or change to 'CA' for CA only)
                                backbone_atoms.append(atom)
        return backbone_atoms

    def compute_rmsd(ref_atoms, target_atoms):
        """Compute RMSD between two sets of atoms."""
        assert len(ref_atoms) == len(
            target_atoms
        ), "Atom sets must have the same length"

        # Calculate RMSD manually
        diff = np.array(
            [atom1.coord - atom2.coord for atom1, atom2 in zip(ref_atoms, target_atoms)]
        )
        return np.sqrt((diff**2).sum() / len(ref_atoms))

    parser = PDBParser(QUIET=True)

    # Load the reference structure
    ref_structure = parser.get_structure("reference", reference_pdb)

    # Extract reference backbone atoms
    ref_atoms = extract_backbone_atoms(ref_structure)

    # Load the target structure
    target_structure = parser.get_structure("target", target_pdb)

    # Extract target backbone atoms
    target_atoms = extract_backbone_atoms(target_structure)

    # Align the structures (using Biopython's Superimposer)
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)  # Set atoms for alignment
    super_imposer.apply(target_structure.get_atoms())  # Apply the transformation

    # Compute the RMSD between the aligned atoms
    rmsd_value = compute_rmsd(ref_atoms, target_atoms)

    return rmsd_value


def count_clashes(structure_path, clash_cutoff=0.65):
    # read
    sloppyparser = PDB.PDBParser()
    structure = sloppyparser.get_structure("struct", structure_path)
    # Atomic radii for various atom types.
    # You can comment out the ones you don't care about or add new ones
    atom_radii = {
        #    "H": 1.20,  # Who cares about hydrogen??
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "F": 1.47,
        "P": 1.80,
        "CL": 1.75,
        "MG": 1.73,
    }
    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii
        for j in atom_radii
    }
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(
            np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values())
        )
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2

    def check_ensemble(self, ensemble: np.array) -> bool:

        ## check ensemble requirements

        return True

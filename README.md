# AF2χ for localColabFold

## 📝 About

Implementation of AF2χ ([Cagiada M., Thomasen F.E., et al., bioRxiv 2025](https://www.biorxiv.com)) using [localColabFold](https://github.com/YoshitakaMo/localcolabfold) as base code for AF2 ([Jumper J., et al., Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)).

AF2χ is a tool to predict side-chain heterogeneity using AlphaFold2 and its internal side-chain representations. AF2χ outputs side-chain χ-angle distributions and structural ensembles around the predicted AF2 structure.

The code in this repository allows you to run AF2χ by downloading localColabFold, patching its code, and adding the additional AF2χ functionality to the original localColabFold implementation.

---

AF2χ is currently available for the Linux distribution of localColabFold, using a stable forked repository of ColabFold [v1.5.5](https://github.com/matteo-cagiada/ColabFold-sc) (`commit: fdf3b235b88746681c46ea12bcded76ecf8e1f76` - July 2024) and Alphafold [2.3.7](https://pypi.org/project/alphafold-colabfold).

### 📂 Repository layout

| Path | Purpose |
| --- | --- |
| `install_colabbatch_linux.sh` | installs the localColabFold version AF2χ is built on |
| `patcher_colabfold_linux.sh` | copies the AF2χ code and parameters into that installation |
| `src/` | the AF2χ code and rotamer parameters applied by the patcher |
| `tools/traj2af2chi.py` | builds a template ensemble (mmCIF) from a trajectory or PDB files |
| `tools/environment.yml`, `install_af2chi_tools.sh` | small separate environment for the tools |
| `Dockerfile`, `Dockerfile-fat` | container recipes |

## ✨ What's new (Sept-Oct 2026 update)

This release adds three things over the first version of AF2χ:

- **Complexes are now supported.** Side-chain prediction and ensemble generation work on multi-chain systems, not only monomers. Pass a complex the same way you would in ColabFold (chains separated by `:` in the FASTA, or a multi-chain template), and AF2χ handles the per-chain backbones and χ-angle bookkeeping for you.

- **Ensemble-based generation (`--generate-from-templates`).** Instead of building the structural ensemble around a single AF2 backbone, AF2χ can take a *conformational ensemble* of templates and sample backbones from it, reweighting the side-chain distributions on top. This lets you propagate backbone heterogeneity (from MD, NMR models, or multiple crystal forms) into the side-chain ensemble. See the `--af2chi-ensemble` preset below.

- **Stricter clash checking.** Generated structures are now screened with a more strict heavy-atom clash detector that ignores pairs fixed by peptide geometry (1,2 bonds and 1,3 neighbours, including the proline ring) and disulfide-bonded SG pairs, so real side-chain clashes are no longer masked by bonded contacts.

- **One-flag presets.** `--af2chi-backbone` and `--af2chi-ensemble` expand into the full, tested option sets for the two standard AF2χ configurations. See [Standard configurations](#-standard-configurations).

- **A trajectory converter.** `tools/traj2af2chi.py` turns an MD trajectory, a multi-model PDB, or a folder of PDB files into an AF2-compatible mmCIF ensemble in one step, ready for `--af2chi-ensemble`. See [Helper tools](#-helper-tools-building-a-template-ensemble).

## ⚙️ Installing AF2χ

### 🔹 Install localColabFold

To start, clone the repository on your local machine and navigate in the repository directory:

```bash
git clone https://github.com/matteo-cagiada/AF2chi_localcolabfold.git

cd AF2chi_localcolabfold
```

Next, you need to install a localColabFold version compatible with AF2χ. We provide an installation script `install_colabbatch_linux.sh` in the repository, which installs our tested version of localColabFold. The script is a modified version of the original installation script from the localColabFold repository, with adjustments to dependencies to maximise compatibility.

**N.B.:** LocalColabFold works with **CUDA >= 12.0**. If you encounter dependency issues, refer to the [localColabFold](https://github.com/YoshitakaMo/localcolabfold) documentation for troubleshooting.

```bash
# Use install_colabbatch_linux.sh to install localColabFold
./install_colabbatch_linux.sh
```

#### 📁 Installation Directory

**By default** `install_colabbatch_linux.sh` installs localColabFold in the directory where the script is executed. If you prefer a different location, move the script to your desired directory before running it.

---

### 🔹 Applying the AF2χ Patch

#### ✅ If localColabFold is installed in the default `af2chi_localcolabfold` directory:

```bash
# Apply the patch to the default installed localColabFold version
./patcher_colabfold_linux.sh
```

#### ✅ If localColabFold is installed in a different directory:

Run the patcher script and provide the path to the localColabFold folder (localcolabfold) as an argument:

```bash
# Apply the patch to localColabFold in a custom location
./patcher_colabfold_linux.sh <path-to-colab-conda>
```

##### Example Usage

If localColabFold is installed in `/users/your_username/home/bin/`, the command line would be:

```bash
./patcher_colabfold_linux.sh /users/your_username/home/bin/localcolabfold/
```

The patcher will replace the file in the localcolabfold installation and add the AF2χ data dependencies and parameters.

---

## 🚀 Inference with AF2χ

AF2χ inference is similar to the original localColabfold implementation.

➡️ **Activate conda enviroment:**
You need first to make the inference command `colabfold_batch` available: to do this you can either:

1. add localColabFold to the enviromental variable list:

```bash
# For bash or zsh
# e.g. export PATH="/<path_to_folder>/localcolabfold/colabfold-conda/bin:$PATH"
export PATH="/<path_to_folder>/localcolabfold/colabfold-conda/bin:$PATH"
```

2. or activate the localColabFold enviroment directly with conda :

```bash
conda activate  /<path_to_folder>/localcolabfold/colabfold-conda
```

You can now run the main localColabFold inference script `colabfold_batch`.

➡️ **Running inference**

`colabfold_batch` provides many options. To see all the options available use the help command:

```bash
colabfold_batch --help
```

AF2χ options are displayed in the AF2chi section, here reported:

```
AF2chi:

  --af2chi              run af2chi to predict sidechain populations and generate a structural
                        ensemble with sidechain predictions (default: False)
  --af2chi-backbone TEMPLATE_PATH
                        Standard AF2chi run on a single backbone (preset, see below)
                        (default: None)
  --af2chi-ensemble TEMPLATE_PATH
                        Standard AF2chi run reweighting a template ensemble (preset, see below)
                        (default: None)
  --no-reweight         run af2chis production on prior library, don't apply re-weighting
                        (default: False)
  --no-ensemble         do not create ensemble of pdb with sidechain predictions, only save the
                        sidechain chi distributions (default: False)
  --no-save-distributions
                        do not save the sidechain chi distributions (default: False)
  --struct-weight STRUCT_WEIGHT
                        run af2sidechains with specified struct-weight (0.85 is default)
                        (default: 0.85)
  --n-struct-ensemble N_STRUCT_ENSEMBLE
                        number of structures to generate in the af2chi ensemble (default: 100)
  --custom-prior CUSTOM_PRIOR
                        path to custom distribution to use as prior in af2sidechains
                        (default: None)
  --generate-from-templates
                        Use templates to generate backbones for af2chi ensemble.
                        (default: False)
  --clash-threshold CLASH_THRESHOLD
                        Heavy-atom clash cutoff: atoms clash if distance < FACTOR*(r_i+r_j).
                        Higher = stricter. Usable range: 0.65-0.80. (default: 0.8)

```

The different options allow you to run the AF2χ pipeline either partially or fully. You can also adjust several parameters, including the number of output structures in the final ensemble.

---

## 🎛 Standard configurations

We tested AF2χ in three configurations. The last two have a **one-flag preset** that expands to the full option set, so you only need to give the template folder.

### 1. AF2χ with standard AF2 inference

This setup uses full MSA and no structural templates as input to the model. It is recommended when the native structure of your protein is unknown.

```bash
colabfold_batch --af2chi <input_fasta> <output_folder>
```

### 2. AF2χ around a single template backbone — `--af2chi-backbone`

Based on [Roney & Ovchinnikov, 2022](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.238101), this configuration uses the query sequence together with a custom structure template and no MSA. It samples χ-angle distributions and generates a structural ensemble around the input template, and is the recommended setup when you have a structure you want to describe.

```bash
colabfold_batch --af2chi-backbone ../templates/ <input_fasta> <output_folder>
```

which is equivalent to:

```bash
colabfold_batch --af2chi --templates --custom-template-path ../templates/ \
  --msa-mode single_sequence --model-order 1,2 --n-struct-ensemble 100 \
  <input_fasta> <output_folder>
```

### 3. AF2χ reweighting a template ensemble — `--af2chi-ensemble`

Here the template folder contains a **conformational ensemble** rather than a single structure. Backbones are drawn from the ensemble and the side-chain distributions are reweighted on top of them, so backbone heterogeneity is carried into the final side-chain ensemble.

```bash
colabfold_batch --af2chi-ensemble ../templates/ <input_fasta> <output_folder>
```

which is equivalent to:

```bash
colabfold_batch --af2chi --templates --custom-template-path ../templates/ \
  --msa-mode single_sequence --model-order 1,2 --generate-from-templates \
  --n-struct-ensemble 100 <input_fasta> <output_folder>
```

### 🔧 Customising a preset

Everything a preset sets is a **default**: pass the option yourself and your value is used instead. For example, a smaller ensemble:

```bash
colabfold_batch --af2chi-ensemble ../templates/ --n-struct-ensemble 50 \
  <input_fasta> <output_folder>
```

The presets always enable `--af2chi`, `--templates` and (for `--af2chi-ensemble`) `--generate-from-templates`, and they own the template path, so `--custom-template-path` and `--pdb-hit-file` cannot be combined with them. Everything else — `--msa-mode`, `--model-order`, `--n-struct-ensemble`, `--struct-weight`, `--clash-threshold`, `--no-reweight`, … — is yours to override. Every override is written to `log.txt`.

⚠️ **Options are not adjusted for each other.** If you change `--msa-mode` away from `single_sequence`, the model order stays at the preset value `1,2` and AF2χ only warns you about it. Models 1 and 2 are the template-aware ones; with a real MSA you usually want `--model-order 3,4,5`:

```bash
colabfold_batch --af2chi-backbone ../templates/ \
  --msa-mode mmseqs2_uniref_env --model-order 3,4,5 <input_fasta> <output_folder>
```

### ⚠️ Template input

1. AF2 accepts only mmCIF (.cif) files as input templates. You can download .cif files directly from the RCSB PDB, or convert your .pdb files using:

   - [pdb-extract](https://pdb-extract.wwpdb.org/) (official)
   - [Neurosnap](https://neurosnap.ai/service/PDB-mmCIF%20Converter)
   - [PDBtools](https://www.bonvinlab.org/pdb-tools/)

2. **Folder & naming:** the template files must be placed inside their own folder and named using 4 lowercase letters/numbers, following classic PDB naming conventions.

3. **Multiple templates & complexes:** there are no restrictions on the number of input templates. AF2 will automatically use any compatible structure that aligns with the query sequence. Complex structures can also be used as templates.

4. **For `--af2chi-ensemble`:** every conformer of your ensemble goes in the same folder, one file per conformer, each with its own 4-character name. AF2χ reports how many backbones it loaded at the start of ensemble generation — check that number matches your ensemble size, since a conformer that fails to parse is skipped with a warning.

5. **Building the folder automatically:** if your ensemble comes from MD, from NMR models, or from any set of PDB files, use `tools/traj2af2chi.py` instead of converting by hand — see the next section.

### 🧬 Complexes

AF2χ now supports multi-chain systems. Provide the complex as you would in ColabFold (chains separated by `:` in the FASTA file) and, if you are using templates, a multi-chain template structure. Both presets work on complexes. Refer to the ColabFold documentation for the details of complex input formatting.

---

## 🧰 Helper tools: building a template ensemble

`tools/traj2af2chi.py` converts a trajectory, a multi-model PDB, or a folder of PDB files into the mmCIF ensemble that `--af2chi-ensemble` expects. It handles the AF2-specific requirements for you: 4-character lowercase names, the metadata block AF2's template featurizer needs, and the residue naming that MD force fields get wrong from AF2's point of view.

### 🔹 Installing the tools environment

The converter needs **MDAnalysis** for trajectory input. Keep it out of the AF2χ environment: MDAnalysis brings its own numpy/scipy constraints, and the AF2χ environment has a carefully pinned jax / tensorflow / flax stack that is not worth disturbing for a script that runs once per ensemble.

```bash
# creates ./af2chi-tools-env with python, MDAnalysis and Biopython
./install_af2chi_tools.sh

# or, if you prefer to drive conda yourself
conda env create -f tools/environment.yml
```

The installer uses whatever it finds (micromamba, mamba or conda) and downloads a local micromamba if there is none. It needs no root access and does not touch your base environment.

**PDB input needs nothing extra.** MDAnalysis is imported only when you pass a trajectory, so `-p` works inside the AF2χ environment itself, which already has Biopython.

### 🔹 Usage

```bash
conda activate ./af2chi-tools-env

# every 10th frame of a trajectory, protein only, frames superimposed
python tools/traj2af2chi.py -x traj.xtc -t topol.tpr -s 10 --align -o templates/

# a frame range
python tools/traj2af2chi.py -x traj.xtc -t topol.gro --start 1000 --stop 2000 -s 5 -o templates/

# NMR models or any multi-model PDB
python tools/traj2af2chi.py -p models.pdb -o templates/

# a folder of single-structure PDB files
python tools/traj2af2chi.py -p pdbs/ -o templates/
```

Then run AF2χ on the result:

```bash
colabfold_batch --af2chi-ensemble templates/ input.fasta results/
```

Main options:

| Option | Meaning |
| --- | --- |
| `-x/--traj`, `-t/--top` | trajectory and topology (`.xtc/.trr/.dcd`, `.tpr/.gro/.pdb/.psf`) |
| `-p/--pdb` | PDB file (single or multi-model) or directory of PDB files |
| `-o/--output` | output directory for the mmCIF ensemble (default `templates/`) |
| `--start`, `--stop`, `-s/--step`, `--frames` | frame selection |
| `--select` | MDAnalysis selection string (default `protein`) |
| `--align` | superimpose all frames on the first selected frame (CA atoms) |
| `--prefix` | first character of the output names (default `x` → `x001.cif`) |
| `--no-resname-fix` | keep MD residue names as they are |
| `--overwrite` | write into a directory that already holds `.cif` files |

### 🔹 What it handles for you

- **MD residue names.** `HID`, `HIE`, `HIP`, `HSD`, `HSE`, `CYX`, `ASH`, `GLH`, `LYN`, `MSE` and friends are renamed to their standard equivalents in the coordinates, not just in the sequence header — AF2's template parser reads the coordinate residue names, and an unknown one costs you the template. Selenomethionine's `SE` atom becomes `SD`, and capping groups (`ACE`, `NME`, `NMA`, `NH2`) are dropped. Every change is reported once.
- **Complexes.** Identical chains are collapsed into a single mmCIF entity, different chains get their own, as AF2 expects. If your topology has no chain IDs, the script warns you, since AF2 would otherwise read a multi-chain system as one long chain.
- **Consistency checks.** It refuses to write into a directory that already contains `.cif` files (AF2χ would use those as templates too), warns if a member has a different sequence from the first one, and warns above 200 templates, where the template search step starts to dominate the runtime.

⚠️ `--align` loads the selected frames into memory. For long trajectories, subsample with `--step` first. Alignment is cosmetic as far as AF2χ is concerned — the ensemble is valid either way — but it makes the output easier to inspect.

⚠️ Names are limited to 999 structures (`x001`–`x999`), which is far more than a useful template ensemble. Use `--step` to subsample.

---

## 🧬 Output of AF2χ

AF2χ generates χ-angle distributions and then samples from these distributions to generate a structural ensemble.

Along with the localColabFold output, the standard output of AF2χ includes:

- The final χ-angle distributions for χ1 and χ2, for the highest ranked model (using AF2 ranking), saved as a dictionary in a JSON file: `{fasta_name}_rank_001_sc_distributions_fitted.json`. For each residue, the χ-angle population is reported as a discrete probability distribution with 36 bins, ranging from 0 to 360 (10-degree binning).

```
Example of {fasta_name}_rank_001_sc_distributions_fitted.json

{
    "chi1": { ### distribution for χ1
        "MET1": [  ### target residues
            2.2316944008858687e-05, ### prob for the 10-degree bin (from 0 to 10 degrees)
            0.00031300707341209516,
            0.003013823938645255,
            0.018280426832940892,
            0.06929143379141825
            ....
            ....
            0.16224556084048378,
            0.2354311056946399,
            0.2115453774166867 ### prob last 10-degree bin (from 350 to 360 degrees)
            ]
            ....
            ....

        "THR102": [ ## second target residue
            0.11728960066637872,
            0.04026974898543387,
            ....
            ....
            0.008523468473728073,
            0.0011102230062327329,
            9.19117342736029e-0]
    }

  "chi2": { ### distribution for χ2
      "MET1":[....]
      .....
    }
}
```

- The structural ensemble, saved as PDB files in the subfolder `sidechain_ensemble`. The standard ensemble size is 100 structures.

Additional AF2χ options available during inference may modify or remove some of these outputs, in particular:

- `--no-ensemble` and `--no-save-distributions` will remove both the final ensemble and the JSON file from the output.

- `--no-reweight` will generate χ-angle populations using the prior distribution, returning these in the dictionary: `{fasta_name}_rank_001_sc_distributions_prior.json`, and generate the structural ensemble using the prior distributions.

### 📄 Reading the log

Ensemble generation reports its settings in `log.txt` (and on screen), which is the quickest way to confirm a run did what you intended:

```
INFO Generating ensemble of 100 structures for 1ubq
INFO Backbone source: template ensemble (37 backbones, sampled uniformly)
INFO Clash threshold: 0.8
...
INFO Ensemble creation complete - 100. structures - acceptance rate: 68.5% - total_tentatives: 146
```

A low acceptance rate means many sampled structures were rejected by the clash or RMSD filters. Lowering `--clash-threshold` (towards `0.65`) relaxes the clash criterion; note that generation stops after `3 × --n-struct-ensemble` attempts, so a very low acceptance rate can produce a smaller ensemble than requested.

---

## 📦 Containerized AF2χ

UPDATE IN PROGRESS (release soon!)

### 🐳 Build AF2χ docker image

1. Make sure that Docker is installed, please follow your operating system instructions
2. Run docker build

```bash
docker build -t af2chi_localcolabfold:latest .
```

### Running AF2χ in docker

This command runs AF2χ on an input file `input.fasta` or directory `$INPUT_DIR` and stores the results in `$OUTPUT_DIR`.
Note that Docker requires that volumes are specified as absolute paths.

The AlphaFold2 parameters should be downloaded and mounted into /cache in the container, in this example command the directory `/path/to/colabfold-cache/cache` is used.

For details, please refer to <https://github.com/sokrypton/ColabFold/wiki/Running-ColabFold-in-Docker> .

```bash
docker run --rm \
    --runtime=nvidia --gpus 1 \
    --env PYTHONUNBUFFERED=TRUE \
    -v /path/to/colabfold-cache/cache:/cache \
    -v "${INPUT_DIR}":/input:ro \
    -v "${OUTPUT_DIR}":/output \
    af2chi_localcolabfold:latest \
    colabfold_batch \
      --af2chi \
      /input/input.fasta \
      /output
```

### Build Apptainer image

Apptainer image can be built from an existing Docker image.

```bash
apptainer build af2chi_localcolabfold_latest.sif  docker-daemon://af2chi_localcolabfold:latest
```

### Running AF2χ in Apptainer

```bash
apptainer run \
    --nv \
    --env PYTHONUNBUFFERED=TRUE \
    -B /path/to/colabfold-cache/cache:/cache \
    -B "${INPUT_DIR}":/input:ro \
    -B "${OUTPUT_DIR}":/output \
    af2chi_localcolabfold_latest.sif \
    colabfold_batch \
      --af2chi \
      /input/input.fasta \
      /output
```

---

## 🛠 Troubleshooting

### 🔹 Common Issues & Fixes

#### ❌ Issue: AF2/openMM GCC Library Errors When Running localColabFold

✅ **Fix:** Ensure that the `colabfold-conda` library path is included in your `LD_LIBRARY_PATH` environment variable. To check, print its current value:

```bash
echo $LD_LIBRARY_PATH
```

If the path is missing, prepend the library location with:

```bash
export LD_LIBRARY_PATH=/<path_to_your_installation>/localcolabfold/colabfold-conda/lib/
```

If issues persist, you may need to install the correct version of **GCC** (the missing library is usually specified in the error message). For more information, refer to the [GCC installation guide](https://gcc.gnu.org/install/).

#### ❌ Issue: GPU Memory Conflicts with Multiple GPUs / Defining a Specific GPU for AF2χ

✅ **Fix:** By default, **AF2χ** (via localColabFold) attempts to utilize all available GPUs, which can cause issues on certain systems. To ensure AF2χ runs on a specific GPU, use the following commands before execution:

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=N  # Replace N with the GPU index (e.g., 0, 1, etc.)
```

#### ❌ Issue: The ensemble is smaller than `--n-struct-ensemble`

✅ **Fix:** Generation stops after `3 × --n-struct-ensemble` attempts. Check the acceptance rate in `log.txt`: if it is low, relax the clash criterion with a lower `--clash-threshold` (e.g. `0.7`), or check that the input template is not strained.

#### ❌ Issue: `--af2chi-ensemble` loads fewer backbones than expected

✅ **Fix:** The number of loaded backbones is logged at the start of ensemble generation. A mismatch means some conformers were skipped: check that every file is a valid mmCIF, named with 4 lowercase characters, and matches the query sequence.

#### ❌ Issue: `traj2af2chi.py` says MDAnalysis is not installed

✅ **Fix:** Trajectory input needs the tools environment: run `./install_af2chi_tools.sh` and activate it. PDB input (`-p`) needs only Biopython and works in the AF2χ environment as it is.

#### ❌ Issue: templates from MD are ignored by AF2, or the sequence does not match

✅ **Fix:** Usually residue naming. Let `traj2af2chi.py` do the conversion rather than converting by hand: it renames force-field residue variants (`HID`, `CYX`, `MSE`, …) to the standard names AF2 expects. If you used `--no-resname-fix`, drop it. Check also that the FASTA sequence matches the template sequence printed by the converter.

#### ❌ Issue: a multi-chain template is read as a single chain

✅ **Fix:** The MD topology has no chain IDs. `traj2af2chi.py` warns when this happens. Set chain IDs before converting, for example in MDAnalysis:

```python
u.select_atoms("segid A").atoms.chainIDs = "A"
u.select_atoms("segid B").atoms.chainIDs = "B"
```

---

## 📝 Reference this work:

If you use our model please cite:

Cagiada, M., Thomasen, F.E., Ovchinnikov S., Deane C.M & Lindorff-Larsen, K. (2025). AF2χ: Predicting protein side-chain rotamer distributions with AlphaFold2. In bioRxiv (p. 2024.05.21.595203). <https://doi.org/10.1101/2024.05.21.595203>

```
@ARTICLE{Cagiada2025-ax,
  title    = "AF2χ: Predicting protein side-chain rotamer distributions with AlphaFold2",
  author   = "Cagiada, Matteo and Thomasen, F. Emil and Ovchinnikov, Sergey and Deane, Charlotte M. and Lindorff-Larsen, Kresten",
  journal  = "bioRxiv",
  pages    = "",
  month    =  ,
  year     =  ,
  language = "en"
```

Also if you use this localColab implementation remember to cite:

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. ColabFold - Making protein folding accessible to all. Nature Methods (2022) doi:10.1038/s41592-022-01488-1
- If you're using AlphaFold, please also cite:
Jumper et al. "Highly accurate protein structure prediction with AlphaFold." Nature (2021) doi: 10.1038/s41586-021-03819-2
- If you're using AlphaFold-multimer, please also cite:
Evans et al. "Protein complex prediction with AlphaFold-Multimer." BioRxiv (2022) doi: 10.1101/2021.10.04.463034v2

## 🙌 Acknowledgements

We are grateful to Yann Vander Meersche and Tatiana Galochkina for providing the data on the computational performance of the ATLAS MD simulations used in our analyses. We thank Alexander Korsunsky for helpful feedback during the revision process and for pointing out corrections that improved the manuscript quality. We thank Daniel Keedy and the Keedy lab (CUNY Advanced Science Research Center) for their PREreview and for their helpful comments on structural validity, which improved the manuscript. The research was supported by the PRISM (Protein Interactions and Stability in Medicine and Genomics) centre funded by the Novo Nordisk Foundation (NNF18OC0033950, to K.L.-L.) and a Novo Nordisk Foundation Postdoctoral Fellowship (NNF23OC0082912; to MC).  We acknowledge access to computational resources via a grant from the Carlsberg Foundation (CF21-0392; to K.L.-L.) and from the ROBUST Resource for Biomolecular Simulations (supported by the Novo Nordisk Foundation grant no.
NF18OC0032608; to K.L-L.).

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📬 Contact

For questions or support with this repository, please use the GitHub issue tab or reach out to us via email:

📧 Matteo Cagiada: <matteo.cagiada@bio.ku.dk>

📧 Emil Thomasen: <fe.thomasen@bio.ku.dk>

📧 Kresten Lindorff-Larsen: <lindorff@bio.ku.dk>

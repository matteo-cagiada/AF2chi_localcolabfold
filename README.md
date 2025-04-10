# AF2χ for localColabFold


## 📝 About
Implementation of AF2χ ([Cagiada M., Thomasen F.E., et al., bioRxiv 2025](https://www.biorxiv.com)) using [localColabFold](https://github.com/YoshitakaMo/localcolabfold) as base code for AF2 ([Jumper J., et al., Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)). 

AF2χ is a tool to predict side-chain heterogeneity using AlphaFold2 and its internal side-chain representations. AF2χ outputs side-chain χ-angle distributions and structural ensembles around the predicted AF2 structure.

The code in this repository allows you to run AF2χ by downloading localColabFold, patching its code, and adding the additional AF2χ functionality to the original localColabFold implementation.

----

AF2χ is currently available for the Linux distribution of localColabFold, using a stable forked repository of ColabFold [v1.5.5](https://github.com/matteo-cagiada/ColabFold-sc) (`commit: fdf3b235b88746681c46ea12bcded76ecf8e1f76` - July 2024) and Alphafold [2.3.7](https://pypi.org/project/alphafold-colabfold).

## ⚙️	Installing AF2χ

### 🔹 Install localColabFold

To start, clone the repository on your local machine and navigate in the repository directory:
```sh
git clone https://github.com/matteo-cagiada/af2chi_localcolabfold.git

cd af2chi_localcolabfold
```
Next, you need to install a localColabFold version compatible with AF2χ. We provide an installation script `install_colabbatch_linux.sh` in the repository, which installs our tested version of localColabFold. The script is a modified version of the original installation script from the localColabFold repository, with adjustments to dependencies to maximise compatibility.

**N.B.:** LocalColabFold works with  **CUDA >= 12.0**. If you encounter dependency issues, refer to the [localColabFold](https://github.com/YoshitakaMo/localcolabfold) documentation for troubleshooting.
```sh
# Use install_colabbatch_linux.sh to install localColabFold
./install_colabbatch_linux.sh
```

#### 📁 Installation Directory

**By default ** `install_colabbatch_linux.sh` installs localColabFold in the directory where the script is executed. If you prefer a different location, move the script to your desired directory before running it.

---

### 🔹 Applying the AF2χ Patch

#### ✅ If localColabFold is installed in the default `af2chi_localcolabfold` directory:
```sh
# Apply the patch to the default installed localColabFold version
./patcher_colabfold_linux.sh
```

#### ✅ If localColabFold is installed in a different directory:
Run the patcher script and provide the path to the localColabFold folder (localcolabfold) as an argument:
```sh
# Apply the patch to localColabFold in a custom location
./patcher_colabfold_linux.sh <path-to-colab-conda>
```

##### Example Usage
If localColabFold is installed in `/users/your_username/home/bin/`, the command line would be:
```sh
./patcher_colabfold_linux.sh /users/your_username/home/bin/localcolabfold/
```

The patcher will replace the file in the localcolabfold installation and add the AF2χ data dependencies and parameters.

----

## 🚀 Inference with AF2χ

AF2χ inference is similar to the original localColabfold implementation.

➡️ Activate conda enviroment:
You need first to make the inference command `colabfold_batch` available: to do this you can either:

1. add localColabFold to the enviromental variable list:
```sh
# For bash or zsh
# e.g. export PATH="/<path_to_folder>/localcolabfold/colabfold-conda/bin:$PATH"
export PATH="/<path_to_folder>/localcolabfold/colabfold-conda/bin:$PATH"
```
2. or activate the localColabFold enviroment directly with conda :
```sh
conda activate  /<path_to_folder>/localcolabfold/colabfold-conda
```

You can now run the main localColabFold inference script `colabfold_batch`

➡️ Running inference

`colabfold_batch` provides many options. To see all the options availble use the help command:

```sh
colabfold_batch --help
```
AF2χ options are display in the AF2chi section, here reported

```text
AF2chi:

  --af2chi              run af2chi to predict sidechain populations and generate a structural ensemble with sidechain predictions (default: False)
  --no-reweight         run af2chis production on prior library, don't apply re-weighting (default: False)
  --no-ensemble         do not create ensemble of pdb with sidechain predictions, only save the sidechain chi distributions (default: False)
  --no-save-distributions
                        do not save the sidechain chi distributions (default: False)
  --struct-weight STRUCT_WEIGHT
                        run af2sidechains with specified struct-weight (0.85 is default) (default: 0.85)
  --n-struct-ensemble N_STRUCT_ENSEMBLE
                        number of structures to generate in the af2chi ensemble (default: 100)
```
The different options allow you to run the AF2χ pipeline either partially or fully. You can also adjust several parameters, including the number of output structures in the final ensemble.

⚠️ **Note:** AF2χ has been tested exclusively on monomeric systems. However, it can also be applied to selected complexes. To run AF2χ on complexes, refer to the ColabFold documentation for instructions on how to input complex structures into the inference pipeline.

We tested AF2χ in two different configurations. You can use either with the following commands:

---

### 1. **AF2χ with Standard AF2 Inference**  
This setup uses full MSA and no structural templates as input to the model. It is recommended when the native structure of your protein is unknown.

You can run AF2χ with standard inference by adding the `--af2chi` flag to your usual localColabFold command:

```sh
colabfold_batch --af2chi <input_fasta> <output_folder>

```
---

### 2. AF2χ with decoy strategy. 
Based on [Roney & Ovchinnikov, 2022](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.238101),the decoy strategy uses the query sequence as input along with a custom structure template and no MSA. We implemented this as the default configuration for AF2χ to enable sampling of dihedral distributions and generation of structural ensembles around any input template.

To run AF2χ with the decoy strategy, use the `--af2chi` flag along with these additional options:

`--af2chi` — to enable template usage

`--custom-template-path` — path to your template structure folder

`--msa-mode single_sequence` — disables MSA generation and uses only the query sequence and template

`--model-order` — runs the two AF2 models trained with templates

```sh
colabfold_batch --af2chi  --templates --custom-template-path ../templates/  \
--msa-mode single_sequence --model-order 1,2  <input_fasta> <output_folder>
```

#### ⚠️ Template input:

1. AF2 accepts only mmCIF (.cif) files as input templates. You can download .cif files directly from the RCSB PDB, or convert your .pdb files using:

- [pdb-extract](https://pdb-extract.wwpdb.org/) (official)

- [Neurosnap](https://neurosnap.ai/service/PDB-mmCIF%20Converter)

- [PDBtools](https://www.bonvinlab.org/pdb-tools/)

2. Folder & Naming: The template file must be placed inside its own folder and named using 4 lowercase letters/numbers, following classic PDB naming conventions.

3. Multiple Templates Support & complexes : There are no restrictions on the number of input templates. AF2 will automatically use any compatible structure that aligns with the query sequence. You can also use complex structures as templates—see the ColabFold documentation for further details.

----

## 🧬	 Output of AF2χ

AF2χ generates χ-angle distributions and then samples from these distributions to generate a structural ensemble.

Along with the localColabFold output, the standard output of AF2χ includes:

- The final χ-angle distributions for χ1 and χ2, for the highest ranked model (using AF2 ranking), saved as a dictionary in a JSON file: `{fasta_name}_rank_001_sc_distributions_fitted.json`. For each residue, the χ-angle population is reported as a discrete probability distribution with 36 bins, ranging from 0 to 360 (10-degree binning).

```text
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

----

## 🛠 Troubleshooting

### 🔹 Common Issues & Fixes

#### ❌ Issue: AF2/openMM GCC Library Errors When Running localColabFold
✅ **Fix:** Ensure that the `colabfold-conda` library path is included in your `LD_LIBRARY_PATH` environment variable. To check, print its current value:
```sh
echo $LD_LIBRARY_PATH
```
If the path is missing, prepend the library location with:
```sh
export LD_LIBRARY_PATH=/<path_to_your_installation>/localcolabfold/colabfold-conda/lib/
```
If issues persist, you may need to install the correct version of **GCC** (the missing library is usually specified in the error message). For more information, refer to the [GCC installation guide](https://gcc.gnu.org/install/).

#### ❌ Issue: GPU Memory Conflicts with Multiple GPUs / Defining a Specific GPU for AF2χ
✅ **Fix:** By default, **AF2χ** (via localColabFold) attempts to utilize all available GPUs, which can cause issues on certain systems. To ensure AF2χ runs on a specific GPU, use the following commands before execution:
```sh
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=N  # Replace N with the GPU index (e.g., 0, 1, etc.)
```
----

## ➡️  Coming soon:
- Refined complex prediction (formatted output)

---- 
## 📝 Reference this work:

If you use our model please cite:

Cagiada, M., Thomasen, F.E., Ovchinnikov S., Deane C.M &  Lindorff-Larsen, K. (2025). AF2χ: Predicting protein side-chain rotamer distributions with AlphaFold2. In bioRxiv (p. 2024.05.21.595203). https://doi.org/10.1101/2024.05.21.595203

```text
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

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. ColabFold - Making protein folding accessible to all. Nature Methods (2022) doi:[ 10.1038/s41592-022-01488-1]( 10.1038/s41592-022-01488-1)
- If you’re using AlphaFold, please also cite:
  Jumper et al. "Highly accurate protein structure prediction with AlphaFold." Nature (2021) doi: [10.1038/s41586-021-03819-2](10.1038/s41586-021-03819-2)
- If you’re using AlphaFold-multimer, please also cite:
  Evans et al. "Protein complex prediction with AlphaFold-Multimer." BioRxiv (2022) doi: [10.1101/2021.10.04.463034v2](10.1101/2021.10.04.463034v2)

## 🙌 Acknowledgements  

The research was supported by the PRISM (Protein Interactions and Stability in Medicine and Genomics) centre funded by the Novo Nordisk Foundation (NNF18OC0033950, to K.L.-L.), a Novo Nordisk Foundation Postdoctoral Fellowship (NNF23OC0082912; to MC). 
We acknowledge access to computational resources via a grant from the Carlsberg Foundation (CF21-0392; to K.L.-L.).

----

## 📜 License  
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📬 Contact  
For questions or support with this repository, please use the GitHub issue tab or reach out us via email:  

📧 Matteo Cagiada: [matteo.cagiada@bio.ku.dk](mailto:matteo.cagiada@bio.ku.dk)

📧 Emil Thomasen [fe.thomasen@bio.ku.dk](mailto:fe.thomasen@bio.ku.dk)  

📧 Kresten Lindorff-Larsen [lindorff@bio.ku.dk](mailto:lindorff@bio.ku.dk)  

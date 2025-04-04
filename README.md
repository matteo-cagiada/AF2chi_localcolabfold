# AF2χ for localColabFold


## 📝 About
Implementation of AF2χ using localColabfold as base code for AF2. AF2χ uses AlphaFold2 and its internal side-chain representations to predict side-chain χ-angle distributions and generates structural ensembles around the predicted AF2 structure
The in this repository download and patch localColabfold (link) to the AF2chi version.


This repository implements AF2χ by patching the localColabFold code, adding the extra set of AF2χ to the original localColabFold implementation.

---

AF2χ is currently available for the Linux distribution of localColabFold.

### 🔹 Installing AF2χ@localColabFold
Follow these steps to install and set up AF2χ:

```sh
# Clone the repository
git clone https://github.com/matteo-cagiada/af2chi_localcolabfold.git

# Navigate into the directory
cd af2chi_localcolabfold
```

### 🔹 Installing localColabFold
First, localColabFold must be installed. It is recommended to have **CUDA 12.4** installed. If you encounter dependency issues, refer to the [localColabFold repository](https://github.com/YoshitakaMo/localcolabfold) for troubleshooting.

```sh
# Use install_colabbatch_linux.sh to install localColabFold
./install_colabbatch_linux.sh
```
This script is a modified version of the installation script from the localColabFold repository, with adjustments for dependency compatibility.

### 🔹 Installation Directory

**By default,** `install_colabbatch_linux.sh` installs localColabFold in the directory where the script is executed. If you prefer a different location, move the script to your desired directory before running it.

### 🔹 Applying the Patch

#### ✅ If localColabFold is installed in the default `af2chi_localcolabfold` directory:
```sh
# Apply the patch to the default installed localColabFold version
./patcher_colabfold_linux.sh
```

#### ✅ If localColabFold is installed in a different directory:
Run the patcher script and provide the path to the localColabFold Conda installation (colabfold-conda) as an argument:
```sh
# Apply the patch to localColabFold in a custom location
./patcher_colabfold_linux.sh <path-to-colab-conda>
```

### 🔹 Example Usage
If localColabFold is installed in `/users/your_username/home/bin/`, the command would be:
```sh
./patcher_colabfold_linux.sh /users/your_username/home/bin/localcolabfold/colabfold-conda
```

The patcher will replace in seconds the file in the localcolabfold installation

---

## 🚀 Examples

Here’s how to use this project:

```sh
# Run the main script
python script.py --option value
```

Example output:
```plaintext
[INFO] Processing data...
[SUCCESS] Output saved to results/
```

For more examples, check out the [Usage Guide](#).

---
## 🛠 Troubleshooting

### 🔹 Common Issues & Fixes

#### ❌ Issue: GCC Library Errors When Running localColabFold
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
---

## 🙌 Acknowledgements  

Thanks to the following for their contributions and inspiration:  
- [Contributor 1](https://github.com/contributor1) - Feature development  
- [Contributor 2](https://github.com/contributor2) - Documentation  
- Open-source libraries such as **NumPy, Pandas, TensorFlow**  

---

## 📜 License  
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📬 Contact  
For questions or support, reach out via:  
📧 [your.email@example.com](mailto:your.email@example.com)  
🔗 [Your Website](https://yourwebsite.com)

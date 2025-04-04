# AF2χ for localColabFold


## 📝 About
Implementation of AF2χ using localColabfold as base code for AF2. AF2χ uses AlphaFold2 and its internal side-chain representations to predict side-chain χ-angle distributions and generates structural ensembles around the predicted AF2 structure
The in this repository download and patch localColabfold (link) to the AF2chi version.


This repository implements AF2χ by patching the localColabFold code, adding the extra set of AF2χ to the original localColabFold implementation.

---

## ⚙️ Installation

AF2χ is currently available for the linux distribution of localColabFold.

To install and set up the AF2χ@localColabfold follow these step:

```sh
# Clone this repository
git clone https://github.com/matteo-cagiada/af2chi_localcolabfold.git

# Navigate into the directory
cd af2chi_localcolabfold

#Use install_colabbatch_linux.sh script to install localColabFold. This is a copy of the installation script from localcolabfold repository, with a couple of modified dependencies.
./install_colabbatch_linux.sh

# Use the patcher on the installed localColabFold version
./patcher_colabfold_linux.sh
```
Extra notes here

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

### Common Issues & Fixes  

#### ❌ Issue: Dependency errors when running the project  
✅ **Fix:** Ensure you have the correct Python version:  
```sh
python --version  # Should be Python 3.x
```
If issues persist, try reinstalling dependencies:  
```sh
pip install --upgrade -r requirements.txt
```

#### ❌ Issue: Permission errors on Unix-based systems  
✅ **Fix:**  
```sh
chmod +x script.py  # Grant execution permissions
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

#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# ==========================================
#  ColabFold Environment Sync Script
#  Author: Your Name
#  Date: $(date +%Y-%m-%d)
#  Description: 
#  This script copies files from a source 
#  directory to an existing destination in 
#  the colab-fold environment.
# ==========================================

echo -e "\e[36m==========================================\e[0m"
echo -e "\e[36m AF2\u03C7 code patcher for localColabFold \e[0m"
echo -e "\e[36m Author: Matteo Cagiada - email matteo.cagiada@bio.ku.dk  \e[0m"
echo -e "\e[36m Date: $(date +%Y-%m-%d) \e[0m"
echo -e "\e[36m Code for localColabFold version 1.5.5 (May,2024 update) \e[0m"
echo -e "\e[36m==========================================\e[0m"


CURRENT_DIR=$(pwd)
DEFAULT_DESTINATION="$CURRENT_DIR/localcolabfold/colabfold-conda"

destination_folder="${1:-$DEFAULT_DESTINATION}"

# Function to display usage help
usage() {
    echo -e "\e[31m[ERROR]\e[0m Usage: $0 [destination_path]"
    echo "If no destination path is provided, the default is $DEFAULT_DESTINATION"
    exit 1
}

echo -e "\e[34m[INFO]\e[0m Current directory: ${CURRENT_DIR}"

# Convert to realpath and check if it was successful
destination_folder=$(realpath "$destination_folder" 2>/dev/null) || {
    echo -e "\e[31m[ERROR]\e[0m Failed to resolve realpath for destination: $destination_folder"
    exit 1
}
source_folder="$CURRENT_DIR/src"

echo -e "\e[34m[INFO]\e[0m Source folder: ${source_folder}"
echo -e "\e[34m[INFO]\e[0m Destination path: ${destination_folder}"


# Check if source folder exists
if [ ! -d "$source_folder" ]; then
    echo -e "\e[31m[ERROR]\e[0m Source folder does not exist: $source_folder"
    echo "Ensure the script is run from its own directory."
    exit 1
fi

# Check if destination path exists
if [ ! -d "$destination_folder" ]; then
    echo -e "\e[31m[ERROR]\e[0m Destination path does not exist: $destination_folder"
    usage
fi

echo -e "\e[34m[INFO]\e[0m Destination path: ${destination_folder}"
# Navigate to the destination directory
cd "$destination_folder" || { echo -e "\e[31m[ERROR]\e[0m Failed to enter destination directory"; exit 1; }

# Find the correct Python version directory dynamically
python_folder=$(find "$destination_folder/lib" -maxdepth 1 -type d -name "python*" | sort -rV | head -n 1)

# Check if a Python directory was found
if [ -z "$python_folder" ]; then
    echo -e "\e[31m[ERROR]\e[0m No Python installation found in $destination_folder/lib"
    exit 1
fi
# Define the site-packages path
site_packages_folder="$python_folder/site-packages"

echo -e "\e[34m[INFO]\e[0m Site-packages folder: ${site_packages_folder}"

# Check if site-packages directory exists
if [ ! -d "$site_packages_folder" ]; then
    echo -e "\e[31m[ERROR]\e[0m Site-packages folder not found in $python_folder"
    exit 1
fi

echo -e "\e[34m[INFO]\e[0m Using site-packages folder: $site_packages_folder"

# Iterate through subdirectories in the source folder
for subdir in "$source_folder"/*/; do
    subdir_name=$(basename "$subdir")
    dest_subdir="$site_packages_folder/$subdir_name"

    echo -e "\e[34m[INFO]\e[0m Checking: ${subdir_name}"

    # Only process if the corresponding subdirectory exists in the destination
    if [ -d "$dest_subdir" ]; then
        echo -e "\e[32m[SUCCESS]\e[0m Processing directory: $subdir_name"
        
        # Copy files and directories recursively
        cp -r "$subdir"* "$dest_subdir/"
        echo -e "\e[32m[SUCCESS]\e[0m Copied contents of $subdir_name to $dest_subdir"
    else
        echo -e "\e[33m[WARNING]\e[0m Skipping $subdir_name: Not found in destination"
    fi
done

echo -e "\e[32m[SUCCESS]\e[0m All files copied successfully."

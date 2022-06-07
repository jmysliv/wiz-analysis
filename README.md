# wiz-analysis

COMPARATIVE STUDY OF EMBEDDING METHODS FOR VARIOUS DATASETS

## Quick start

```bash
# create venv
python -m venv venv

# activate venv
source venv/bin/activate

# Set up Python environment
pip install -r requirements.txt

# Run code
python bin/main.py

# save requirements
pip freeze > requirements.txt
```

## How to download datasets

Datasets are too large to put them in the repository. However you can easily download them from this Google Drive: https://drive.google.com/drive/folders/1Gw6cbUBQeR87U47a3NgzBUpsGNDlmZlJ?usp=sharing

Then remember to place them inside `data` directory.

## Largevis

```bash
# Install required library
brew install gsl
```

Compile and put compiled version inside `Largevis` directory.

```
git clone https://github.com/lferry007/LargeVis.git
cd LargeVis/Linux
g++ LargeVis.cpp main.cpp -o LargeVis -lm -pthread -lgsl -lgslcblas -Ofast -ffast-math
```

## IVHD

```bash
cd viskit
git submodule init
git submodule update
# required tools for viskit cmake
brew install libomp
brew install boost
# build
cmake ./CMakeLists.txt
make
```

## Repo structure

 - `bin/main.py`: Code entry point
 - `conf/conf.py`: Configuration file for project
 - `data`: Directory for datasets
 - `graphs`: Directory for graphs required for IVHD
 - `outputs`: Directory for png files


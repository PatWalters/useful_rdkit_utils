useful_rdkit_utils
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/PatWalters/useful_rdkit_utils/workflows/CI/badge.svg)](https://github.com/PatWalters/useful_rdkit_utils/actions?query=workflow%3ACI)


A motley collection of useful RDKit functions

<img src="nikki.png" alt="Robot Nikki" width="300"/>

### Installation
Requires Python 3.11 or newer.
```bash
pip install useful_rdkit_utils
```

A few pieces need optional dependencies. Install `all` to get everything:
```bash
pip install useful_rdkit_utils[all]
```

| Extra | Brings in | Needed for |
| --- | --- | --- |
| `extras` | click | the `build_ring_dictionary` command line tool |
| `jupyter` | IPython | the `rd_*` helpers that configure RDKit's notebook rendering |
| `all` | click, IPython | both of the above |

### Installing the latest development version
```bash
pip install git+https://github.com/PatWalters/useful_rdkit_utils
```

### Command line tools
Installing the package puts two commands on your `PATH`.

| Command | What it does |
| --- | --- |
| `build_ring_dictionary` | build a ring system frequency dictionary from a ChEMBL chemreps file, or look up the ring systems in a set of molecules against one |
| `silly_walks` | build the Morgan fingerprint count dictionary that `SillyWalks` scores against |

### Documentation

[https://useful-rdkit-utils.rtfd.io](https://useful-rdkit-utils.rtfd.io)

### Demos
The notebooks directory contains Jupyter notebooks that show how to use some of the functions in this collection.

### Development
```bash
git clone https://github.com/PatWalters/useful_rdkit_utils.git
cd useful_rdkit_utils
pip install -e ".[tests,all]"
pytest -m "not network"   # some tests download data; -m "not network" skips them
```

### Contributing
I'm very open to PRs with corrections and/or contributions.

### Copyright

Copyright (c) 2021-2026, Pat Walters

### License

The code in this package is licensed under the MIT License.

### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

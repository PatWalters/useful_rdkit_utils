# Changelog
All notable changes to this project will be documented in this file.

## [0.1.5] - 2021-01-10
### Added
- RDKitDescriptors added to useful_rdkit_utils.py
- smi2mol_with_errors added to useful_rdkit_utils.py
- value_counts_df added to pandas_utils.py
- add_molecule_and_errors added to pandas_utils.py

## [0.2] - 2022-12-03
### Added
- Modified fingerprint generation routines to accept smiles
- Add RingSystemFinder class for identifying ring systems
- Add RingSystemLookup class for looking up ring systems in ChEMBL
- Added seaborn_utils.py to set seaborn defaults

## [1.00] - 2026-06-01
### Added
- Added sali.py for SALI structure-activity landscape / activity cliff analysis
- Added model_comparison.py with WrapperFactory, get_performance_stats, and Tukey HSD plotting (make_tukey_plot, plot_r2_mae, plot_tukey_stats)
- Added descriptor_preprocessor.py for descriptor preprocessing
- Added count-based fingerprint generation to descriptors.py
- Added tests for model_comparison, sali, scaffold_finder, and reactions
- Added documentation pages for sali, model_comparison, and descriptor_preprocessor

### Changed
- Added matplotlib and statsmodels as dependencies
- Removed dataclasses from requirements (built in since Python 3.7)
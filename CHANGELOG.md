# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Changed
- Removed the umap-learn dependency: get_t_sne_clusters() in optional.py now uses a two-dimensional t-SNE
  embedding (scikit-learn) instead of UMAP. get_umap_clusters() is kept as a backward-compatible alias.
  The extras install no longer includes umap-learn.
- cross_validate() now performs genuine nested cross-validation (inner CV runs on the outer training set only;
  models are refit and evaluated on the outer test fold) instead of repeating full-data inner CV. Fold labels are
  unique per outer round.
- Renamed the NumRgroupgs column of generate_fragments() to NumRgroups (typo; breaking for consumers of that
  column name).
- Renamed the cutoff parameter of taylor_butina_clustering() and get_butina_clusters() to dist_cutoff to make
  clear it is a distance (1 - Tanimoto similarity), and added range validation.
- Modules now define __all__, so importing useful_rdkit_utils no longer re-exports third-party modules
  (np, pd, plt, Chem, ...).
### Fixed
- bootstrap_confidence_interval() now bootstraps the supplied stat_function instead of hard-coding roc_auc_score.
- pearson_confidence() now uses the two-sided z-score for the requested confidence interval.
- RDKitProperties.pandas_smiles() no longer wraps the column names in an extra list.
- get_scaffold() now accepts RDKit Molecules (it only worked with SMILES and crashed via MurckoScaffoldSmiles)
  and handles molecules with no scaffold.
- mol_to_3D_view() now uses the requested style instead of hard-coding "stick".
- smi2mol_with_errors() now actually captures RDKit parse errors (RDKit writes to file descriptor 2, which
  sys.stderr redirection no longer intercepts in current RDKit versions).
- SillyWalks no longer loads a hard-coded, machine-specific SMILES file on construction; added load_json_dict()
  so a saved count dictionary round-trips with integer fingerprint-bit keys.
- get_unit_multiplier() raises a ValueError for unsupported units.
- REOS.read_rules() now works when active_rules is None (use all rules) instead of crashing.
- RingSystemLookup.process_mol() no longer mutates the caller's molecule (stereo removal happens on a copy).
- enumerate_library_sample() stops when every reagent combination has been tried instead of looping forever.
- plot_r2_mae() and plot_tukey_stats() no longer close the figures they return (they are now displayable in
  notebooks).
- plot_properties() no longer crashes on a one-column DataFrame and returns the Figure.
- mcs_rmsd() returns float("inf") when the molecules share no common substructure, instead of a magic 1e6.
- refine_conformers() raises a clear error when conformers lack an Energy property.
- plot_tukey_plot() -> make_tukey_plot() now honours the method_col argument.
- Tests use the repo data files instead of downloading from GitHub, and no longer write artifacts to the
  working directory.
- Fixed the DescriptorPreprocessor doctest expected shape.

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
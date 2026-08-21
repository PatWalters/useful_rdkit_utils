# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Python 3.13 to the CI matrix, and 3.11/3.12/3.13 to the package classifiers. requires-python stays at >=3.11:
  the code needs nothing newer than 3.9 syntactically, and 3.11 is still supported upstream, so raising the floor
  would drop users for no reason. On 3.11, pip resolves slightly older numpy and scipy, whose current releases
  require 3.12.
- A `jupyter` extra providing IPython, and an `all` extra pulling in every optional dependency. The rd_* helpers
  in jupyter_utils configure RDKit's notebook rendering, which RDKit implements on top of IPython, so all but one
  of them raised ModuleNotFoundError on an install that did not happen to have IPython present.
- Console script entry points for the two CLIs that already existed but were unreachable after installation:
  `build_ring_dictionary` and `silly_walks`.
- A `docs` extra holding the Sphinx toolchain, and a `network` pytest marker for the tests that download data
  (all of tests/test_reos.py). Run `pytest -m "not network"` to skip them.
- refine_conformers() takes a use_symmetry flag. When True, conformers are compared with a symmetry-aware
  RMSD (rdMolAlign.GetBestRMS) so that conformers differing only by a symmetric relabelling -- a flipped
  phenyl, a rotated methyl -- are recognised as redundant rather than kept as distinct. Defaults to False,
  which preserves the existing symmetry-blind comparison.
### Changed
- get_random_clusters() and get_butina_clusters() return numpy arrays, matching get_bemis_murcko_clusters()
  and get_kmeans_clusters(). Two of the four returned lists, which is what tripped the pd.unique() deprecation.
- reaction_demo() and scaffold_finder's main() are no longer exported. Both read data files that are not
  distributed with the package, so neither could run as imported, and main() was reaching the public namespace
  as the generically named uru.main. Both remain in their modules as worked examples.
- Dropped the pyarrow dependency; no module in the package imports it.
- Removed useful_rdkit_utils/data/ring_systems/chembl_ring_systems.parquet. It was ~1 MB of package data that
  no code referenced, and it was a stale snapshot (49,769 ring systems against the 55,146 in the CSV that
  RingSystemLookup actually downloads), so wiring it up would have served outdated frequencies.
- Removed requirements.txt. Its only consumer was .readthedocs.yaml, and it had drifted out of sync with
  pyproject.toml, omitting matplotlib, scipy and statsmodels, so installing from it produced an environment that
  could not import the package. .readthedocs.yaml now installs the package itself with the docs and extras
  extras, which makes pyproject.toml the single place dependencies are declared.
- .readthedocs.yaml builds on ubuntu-24.04 with Python 3.12, replacing ubuntu-20.04 and Python 3.11.
- MANIFEST.in no longer references versioneer.py, which the move to hatchling removed.
- The notebooks that map conformer generation over a dataframe wrap the call, so that a molecule MMFF cannot
  parameterize is recorded as None and dropped rather than stopping the batch. This is the behaviour they had
  when gen_conformers() returned None; ChEMBL drugs such as cisplatin and bortezomib reach that path.
- gen_conformers() raises on failure rather than printing and returning None, so a genuine error (a molecule
  MMFF cannot parameterize, for instance) can be told apart from a small conformer count. Embedding fewer
  conformers than requested now warns and returns them instead of discarding the lot. The return annotation
  changes from Optional[Mol] to Mol.
- REOS.read_rules() reads the file named by its rules_file argument, which it previously ignored in favour of
  the file loaded in __init__.
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
- REOS.set_min_priority() no longer raises UndefinedVariableError. It referenced an `active_rules` name that
  existed in no scope; the selected rule sets are now recorded on the instance (REOS.active_rules) and the
  priority filter resets from them, so repeated calls are not cumulative.
- plot_properties() no longer raises AttributeError for a DataFrame with more than one column. subplots() was
  called with squeeze=False, which returns a (1, n) array, but the axes were indexed as if the array were 1-D.
- refine_conformers() now addresses conformers by their RDKit conformer id rather than by list position. Ids
  are not required to be contiguous or to start at zero, so molecules whose conformers were previously trimmed
  (including by an earlier call to refine_conformers) no longer raise IndexError or silently drop the wrong
  conformers.
- refine_conformers() no longer moves the coordinates of the conformers it keeps. Both RMSD functions align
  the conformers they measure, so a molecule came back rototranslated by as much as 10 A. RMSD is now measured
  against a throwaway copy.
- get_molecules_with_scaffold() builds its molecules from the fragment table rather than from a "mol" column in
  activity_df. It tested activity_df for that column but assigned to the match table, so when activity_df did
  supply one, any scaffold match without an activity row reached RGroupDecompose as NaN and failed with an
  opaque Boost conversion error.
- find_scaffolds() reports the columns it needs by name instead of failing with a bare KeyError. The required
  "mol" column is now documented alongside smiles_col and name_col.
- boxplot_base64_image() takes a tuple as the default x_lim instead of a mutable list.
- refine_conformers() no longer discards conformers that are not redundant. RMSD filtering compared every
  conformer against every other one, so a conformer was removed for resembling one that had itself been
  removed: for conformers A-B-C where A and B are similar and B and C are similar but A and C are not, only A
  survived. Each conformer is now compared against the conformers that were kept, so C is retained.
- refine_conformers() keeps the lowest-energy member of a redundant group. Previously the survivor was
  whichever conformer had the lower id, which is unrelated to energy, so a redundant pair could keep the
  higher-energy conformer and discard the lower-energy one.
- clean_descriptors() and clean_and_scale_descriptors() now accept a pandas DataFrame, which is the type
  clean_and_scale_descriptors() has always declared. A DataFrame is returned as a DataFrame with the surviving
  column labels intact; an ndarray is still returned as an ndarray.
- SillyWalks is now exported from the package (`uru.SillyWalks`). silly_walks was the only module missing from
  the imports in __init__.py.
- The jupyter_utils helpers raise an ImportError naming the extra to install when IPython is missing, instead of
  surfacing a bare ModuleNotFoundError from inside RDKit. The import is also deferred, so importing the module
  (as Sphinx autodoc does) no longer needs IPython at all.
- The build_ring_dictionary command reports that it needs click, and which extra provides it, instead of failing
  with ModuleNotFoundError. click is an optional dependency but the console script is installed unconditionally.
- enumerate_library_sample() no longer loops forever when two reagents share a "_Name". Combinations were
  tracked by the joined reagent name while the termination guard counted reagent combinations, so with
  duplicate names the two could never converge. Combinations are now tracked by reagent position, which also
  means distinct reagents that share a name are all reachable instead of being skipped as already used.
- enumerate_library_sample()'s progress bar advances once per product. It advanced by 100 whenever
  `count % 100 == 0`, so it never moved for targets under 100 and jumped repeatedly while the count was 0.
- enumerate_library() and enumerate_library_sample() sanitize products with catchErrors, so a product that
  cannot be sanitized is skipped (with a warning from enumerate_library) instead of aborting the enumeration.
  The previous `res == SANITIZE_NONE` check was dead code, because SanitizeMol raises by default.
- REOS.process_mol() returns None for a None molecule instead of raising AttributeError, so a single structure
  that failed to parse no longer aborts a whole pandas_mols() run.
- REOS.read_rules() raises ValueError on unparsable SMARTS instead of calling sys.exit(1), which killed the
  host process (and, in a notebook, the kernel).
- REOS.set_active_rule_sets() and read_rules() accept a single rule set name as a bare string, not only a list.
  A string would otherwise be stored character by character, breaking a later set_min_priority() call.
- REOS.set_active_rule_sets() validates the requested rule set names and copies the result, matching
  read_rules(); it previously left an empty selection and a view that triggered SettingWithCopyWarning.
- calculate_sali() drops SMILES that will not parse (with a warning) instead of handing None to the
  fingerprint generator, which aborted the whole O(n^2) calculation.
- get_min_ring_frequency() no longer sorts the caller's list in place, and returns a list on both paths.
- cross_validate() coerces group assignments to an array before calling pd.unique(). Passing a plain list is
  deprecated in pandas and raises in a future release; group_func is user supplied, so the coercion is done at
  the call site.
- split_dataframe() no longer appends an empty chunk when the row count is an exact multiple of chunk_size,
  and rejects a chunk_size below 1.
- RDKitDescriptors validates desc_names at construction, naming the unknown descriptors, rather than failing
  later with a bare KeyError. It also evaluates only the requested descriptors instead of calculating the full
  ~200-descriptor set and selecting from it, which makes a small subset roughly 200x faster.
- SillyWalks.score() returns 0 for a molecule with no fingerprint bits (an empty SMILES) instead of raising
  ZeroDivisionError.
- get_largest_fragment() and generate_fragments() raise a clear ValueError for None or an atomless molecule,
  instead of IndexError and ZeroDivisionError respectively.
- Model wrappers built by WrapperFactory no longer write an "fp" column into the DataFrame they are given, and
  report a missing SMILES by name rather than failing inside np.stack.
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
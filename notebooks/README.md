# useful_rdkit_utils — notebooks

Worked examples for the [`useful_rdkit_utils`](https://github.com/PatWalters/useful_rdkit_utils) package. Each notebook is self-contained — most pull their input data directly from a URL — so you can open any of them and run top-to-bottom after `pip install useful_rdkit_utils`.

## Cheminformatics demos

Quick tours of the core RDKit-flavored utilities in the package.

| Notebook | What it shows |
|----------|---------------|
| [demo_useful_rdkit_utils.ipynb](demo_useful_rdkit_utils.ipynb) | Grab-bag tour: 3D structure generation, 3D viewer, descriptor calculation, fingerprints, ring system identification, and more. Start here if you're new to the package. |
| [demo_descriptors.ipynb](demo_descriptors.ipynb) | RDKit physicochemical/topological descriptors via `RDKitDescriptors`, including `DescriptorPreprocessor` for imputation and scaling. |
| [demo_ring_systems.ipynb](demo_ring_systems.ipynb) | Identify and look up *precedented* ring systems in a molecule. Useful for flagging unusual scaffolds. |
| [demo_REOS.ipynb](demo_REOS.ipynb) | Rapid Elimination of Swill (REOS) functional-group / structural-alert filters. |
| [explore_functional_group_filters.ipynb](explore_functional_group_filters.ipynb) | Visualize the SMARTS patterns behind the functional-group filter sets. |
| [parallel_conformer_generation.ipynb](parallel_conformer_generation.ipynb) | Generate 3D conformers in parallel across multiple cores. |

## Modeling and cross-validation

End-to-end ML workflows that build on `uru.cross_validate` and `uru.WrapperFactory`.

| Notebook | What it shows |
|----------|---------------|
| [demo_cross_validate.ipynb](demo_cross_validate.ipynb) | Full walkthrough of `uru.cross_validate`: nested CV with pluggable descriptors, models, and splitters (random / scaffold / Butina cluster). |
| [model_comparison.ipynb](model_comparison.ipynb) | Compare LightGBM, XGBoost, and Ridge across all six Biogen public ADME endpoints under Butina-clustered CV, with R²/MAE summaries and Tukey HSD pairwise plots. |
| [compare_mordred_rdkit.ipynb](compare_mordred_rdkit.ipynb) | Benchmark RDKit (~200) vs. Mordred (~1,600) 2D descriptor sets, each paired with LightGBM and a feed-forward NN, on the Biogen ADME dataset. |
| [bootstrap_AUC.ipynb](bootstrap_AUC.ipynb) | Bootstrap a confidence interval for classifier AUC. |

## Out-of-distribution detection

Reconstruction-error-based OOD detection using a pretrained SMILES autoencoder.

> ⚠️ These notebooks depend on the [`smiles_ae`](https://zenodo.org/records/18846067) package, a pretrained checkpoint, and pre-split datasets that live outside this repo. Paths near the top of each notebook (`MODEL_DIR`, `file_list`) need to be edited for your machine, and a CUDA GPU is assumed.

| Notebook | What it shows |
|----------|---------------|
| [evaluate_ood.ipynb](evaluate_ood.ipynb) | Fine-tune the autoencoder per dataset, then compare reconstruction-error distributions on in-distribution test vs. OOD splits. |
| [evaluate_ood_v2.ipynb](evaluate_ood_v2.ipynb) | Updated variant of the above. |

## Helper modules

Importable `.py` files used by the notebooks above.

| File | Purpose |
|------|---------|
| [ffnn.py](ffnn.py) | `FFNNRegressor` — a scikit-learn-compatible PyTorch MLP regressor. Used by `compare_mordred_rdkit.ipynb`. |
| [lgbm_wrapper.py](lgbm_wrapper.py) | `LGBMMorganCountWrapper` and `LGBMPropWrapper` — minimal SMILES-in / prediction-out wrappers around LightGBM, illustrating the wrapper pattern. |
| [catboost_wrapper.py](catboost_wrapper.py) | `CatBoostWrapper` — same pattern as above for CatBoost. |
| [sort_and_slice_ecfp_featuriser.py](sort_and_slice_ecfp_featuriser.py) | Reference implementation of the [Sort & Slice](https://arxiv.org/abs/2403.17954) ECFP pooling operator (Dablander et al., 2024), used as an alternative to hash-based folding. |

## Data

- `data/biogen_logS.csv` — solubility subset of the Biogen public ADME data, used by the wrapper-module smoke tests.
- `zinc_sample.smi` — small SMILES sample from ZINC for the conformer-generation demo.

Most notebooks fetch their inputs directly from public URLs (the Biogen ADME CSV on GitHub, etc.), so they should run without any additional setup beyond `pip`.

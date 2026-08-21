"""Regression tests for the defects turned up by the code audit.

Grouped by the module they exercise. Each test names the behaviour that was wrong
so a future change that reintroduces it fails here rather than in a user's script.
"""
import builtins
import importlib.util
import os
import sys
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import useful_rdkit_utils as uru


# --------------------------------------------------------------------------
# reactions.py
# --------------------------------------------------------------------------

AMIDE_RXN = "[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2](=O)N[C:1]"


def _named_mol(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    return mol


@pytest.fixture
def reagent_lol():
    amines = [_named_mol("C" * (i + 1) + "N", f"am{i}") for i in range(5)]
    acids = [_named_mol("C" * (i + 1) + "C(=O)O", f"ac{i}") for i in range(5)]
    return [amines, acids]


def test_enumerate_library_sample_progress_bar_tracks_products(reagent_lol, monkeypatch):
    """The bar advanced on `count % 100 == 0`, so it never moved for small targets."""
    import useful_rdkit_utils.reactions as reactions

    updates = []

    class SpyTqdm(reactions.tqdm):
        def update(self, n=1):
            updates.append(n)
            return super().update(n)

    monkeypatch.setattr(reactions, "tqdm", SpyTqdm)
    rxn = AllChem.ReactionFromSmarts(AMIDE_RXN)
    df = reactions.enumerate_library_sample(rxn, reagent_lol, 10)

    assert len(df) == 10
    assert sum(updates) == len(df), "progress bar total must match the products generated"


def test_enumerate_library_sample_respects_the_requested_count(reagent_lol):
    rxn = AllChem.ReactionFromSmarts(AMIDE_RXN)
    assert len(uru.enumerate_library_sample(rxn, reagent_lol, 7)) == 7


def test_enumerate_library_sample_stops_when_combinations_are_exhausted(reagent_lol):
    """Asking for more products than there are reagent combinations must terminate."""
    rxn = AllChem.ReactionFromSmarts(AMIDE_RXN)
    df = uru.enumerate_library_sample(rxn, reagent_lol, 1000)
    assert 0 < len(df) <= 25


def test_enumerate_library_sample_handles_duplicate_reagent_names():
    """Reagent names are not unique, so combinations must be tracked by position.

    Keying the "already tried" set on the joined reagent name means len(used) can
    never reach the combination count, and the sampling loop spins forever. Run it
    on a daemon thread so a regression fails the test rather than hanging the suite.
    """
    import threading

    rxn = AllChem.ReactionFromSmarts(AMIDE_RXN)
    # three distinct amines that share a name: 3 x 2 = 6 combinations, but 2 names
    amines = [_named_mol(smi, "dup") for smi in ["CN", "CCN", "CCCN"]]
    acids = [_named_mol("CC(=O)O", "ac0"), _named_mol("CCC(=O)O", "ac1")]

    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("df", uru.enumerate_library_sample(rxn, [amines, acids], 6)),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive(), "enumerate_library_sample did not terminate"
    assert len(result["df"]) == 6
    assert result["df"].SMILES.nunique() == 6, "distinct reagents sharing a name were skipped"


def test_enumerate_library_skips_unsanitizable_products(reagent_lol):
    """A bad product is skipped with a warning rather than aborting the enumeration."""
    # this SMARTS builds a five-valent carbon, which cannot be sanitized
    bad_rxn = AllChem.ReactionFromSmarts("[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:1]([C:2])([C:2])([C:2])([C:2])[C:2]")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        products = uru.enumerate_library(bad_rxn, [reagent_lol[0][:2], reagent_lol[1][:2]])
    assert products == []
    assert any("could not be sanitized" in str(w.message) for w in caught)


def test_enumerate_library_returns_all_combinations(reagent_lol):
    rxn = AllChem.ReactionFromSmarts(AMIDE_RXN)
    products = uru.enumerate_library(rxn, [reagent_lol[0][:3], reagent_lol[1][:2]])
    assert len(products) == 6
    assert all(Chem.MolFromSmiles(smi) is not None for smi, _ in products)


# --------------------------------------------------------------------------
# sali.py
# --------------------------------------------------------------------------

def test_calculate_sali_skips_unparsable_smiles():
    """One bad SMILES used to abort the whole O(n^2) calculation."""
    df = pd.DataFrame({"SMILES": ["CCO", "not_a_smiles", "c1ccccc1", "CCN"],
                       "Activity": [1.0, 2.0, 3.0, 4.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = uru.calculate_sali(df)
    assert any("could not be parsed" in str(w.message) for w in caught)
    # three molecules parsed, so three pairs
    assert len(result) == 3
    assert "not_a_smiles" not in set(result.SMILES_1) | set(result.SMILES_2)


def test_calculate_sali_raises_when_nothing_parses():
    df = pd.DataFrame({"SMILES": ["nope", "also_nope"], "Activity": [1.0, 2.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="None of the SMILES"):
            uru.calculate_sali(df)


# --------------------------------------------------------------------------
# geometry.py
# --------------------------------------------------------------------------

def test_gen_conformers_returns_what_it_could_embed(monkeypatch):
    """Embedding fewer conformers than requested warns and returns them, not None.

    ETKDG returns the full count even for rigid molecules, so this branch is driven
    directly rather than through a molecule chosen to be hard to embed.
    """
    import useful_rdkit_utils.geometry as geometry

    real_embed = geometry.AllChem.EmbedMultipleConfs

    def short_embed(mol, numConfs, params):
        return real_embed(mol, numConfs=3, params=params)

    monkeypatch.setattr(geometry.AllChem, "EmbedMultipleConfs", short_embed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mol = geometry.gen_conformers(Chem.MolFromSmiles("CCCCO"), num_confs=50)
    assert mol is not None
    assert mol.GetNumConformers() == 3
    assert any("embedded 3" in str(w.message) for w in caught)


def test_gen_conformers_raises_when_nothing_embeds(monkeypatch):
    """No conformers at all is a failure, not a silent None."""
    import useful_rdkit_utils.geometry as geometry

    monkeypatch.setattr(geometry.AllChem, "EmbedMultipleConfs", lambda mol, numConfs, params: [])
    with pytest.raises(ValueError, match="Could not embed"):
        geometry.gen_conformers(Chem.MolFromSmiles("CCCCO"), num_confs=5)


def test_gen_conformers_conformers_carry_energies():
    mol = uru.gen_conformers(Chem.MolFromSmiles("CCCCO"), num_confs=5)
    assert all(conf.HasProp("Energy") for conf in mol.GetConformers())


def test_gen_conformers_raises_without_mmff_parameters():
    """A molecule MMFF cannot parameterize raises instead of returning None."""
    mol = Chem.MolFromSmiles("B(O)(O)O")
    with pytest.raises(ValueError, match="MMFF parameters"):
        uru.gen_conformers(mol, num_confs=1)


# --------------------------------------------------------------------------
# ring_systems.py
# --------------------------------------------------------------------------

def test_get_min_ring_frequency_does_not_mutate_its_argument():
    ring_list = [("c1ccccc1", 500), ("C1CC1", 2), ("C1CCC1", 30)]
    original = list(ring_list)
    result = uru.get_min_ring_frequency(ring_list)
    assert ring_list == original, "the caller's list was reordered"
    assert result[0] == "C1CC1"
    assert result[1] == 2


def test_get_min_ring_frequency_handles_acyclic():
    assert uru.get_min_ring_frequency([]) == ["", -1]


# --------------------------------------------------------------------------
# split_utils.py
# --------------------------------------------------------------------------

def test_cross_validate_does_not_warn_on_list_valued_groups():
    """pd.unique() on a plain list is deprecated and will raise in a future pandas."""
    from sklearn.linear_model import LinearRegression

    smiles = ["c1ccccc1", "c1ccccc1C", "CCCCCCCC", "CCN", "CCO",
              "CCCl", "c1ccccc1O", "c1ccccc1N", "CCCC", "CCOCC"]
    df = pd.DataFrame({"SMILES": smiles, "y": np.linspace(0, 1, len(smiles))})
    desc = {smi: uru.smi2numpy_fp(smi) for smi in smiles}
    wrapper = uru.WrapperFactory.create_wrapper_class(LinearRegression, desc)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        uru.cross_validate(df, [("lr", wrapper)], "y",
                           # returns a plain list, which is what triggered the warning
                           [("random", uru.get_random_clusters)], n_outer=2, n_inner=2)


# --------------------------------------------------------------------------
# pandas_utils.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_rows,chunk_size,expected", [
    (20, 10, [10, 10]),      # exact multiple: no trailing empty chunk
    (25, 10, [10, 10, 5]),
    (5, 10, [5]),
    (0, 10, []),
    (3, 1, [1, 1, 1]),
])
def test_split_dataframe_chunk_sizes(num_rows, chunk_size, expected):
    df = pd.DataFrame({"a": range(num_rows)})
    assert [len(c) for c in uru.split_dataframe(df, chunk_size=chunk_size)] == expected


def test_split_dataframe_rejects_a_bad_chunk_size():
    with pytest.raises(ValueError):
        uru.split_dataframe(pd.DataFrame({"a": [1, 2]}), chunk_size=0)


def test_value_counts_df():
    df = pd.DataFrame({"x": ["a", "b", "a", "a"]})
    out = uru.value_counts_df(df, "x")
    assert list(out.columns) == ["x", "count"]
    assert out.query("x == 'a'")["count"].iloc[0] == 3


def test_get_dataframe_nans():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    assert len(uru.get_dataframe_nans(df)) == 1


# --------------------------------------------------------------------------
# descriptors.py
# --------------------------------------------------------------------------

def test_rdkit_descriptors_rejects_unknown_names():
    """An unknown name failed late with a bare KeyError naming no alternatives."""
    with pytest.raises(ValueError, match="NotADescriptor"):
        uru.RDKitDescriptors(desc_names=["MolWt", "NotADescriptor"])


def test_rdkit_descriptors_subset_matches_the_full_calculation():
    """Calculating a subset directly must agree with selecting from the full set."""
    smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    subset = ["MolWt", "MolLogP", "TPSA", "NumHDonors"]
    from rdkit.Chem import Descriptors

    reference = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smiles))
    calculated = uru.RDKitDescriptors(desc_names=subset).calc_smiles(smiles)
    assert np.allclose(calculated.astype(float), [reference[x] for x in subset])


def test_rdkit_descriptors_subset_only_evaluates_what_was_asked_for():
    """The subset path must not evaluate all ~200 descriptors."""
    subset = ["MolWt", "TPSA"]
    calls = []
    original = dict(uru.DESCRIPTOR_FUNCTIONS)

    def counting(name, func):
        def wrapped(mol):
            calls.append(name)
            return func(mol)
        return wrapped

    try:
        for name, func in original.items():
            uru.DESCRIPTOR_FUNCTIONS[name] = counting(name, func)
        uru.RDKitDescriptors(desc_names=subset).calc_smiles("CCO")
    finally:
        uru.DESCRIPTOR_FUNCTIONS.clear()
        uru.DESCRIPTOR_FUNCTIONS.update(original)
    assert sorted(calls) == sorted(subset)


def test_rdkit_descriptors_full_set_still_works():
    calc = uru.RDKitDescriptors()
    result = calc.calc_smiles("CCO")
    assert len(result) == len(calc.desc_names)
    assert len(calc.desc_names) > 100


def test_rdkit_descriptors_skip_fragments():
    calc = uru.RDKitDescriptors(skip_fragments=True)
    assert not any("fr_" in name for name in calc.desc_names)


# --------------------------------------------------------------------------
# silly_walks.py
# --------------------------------------------------------------------------

def test_silly_walks_empty_smiles_does_not_divide_by_zero():
    """MolFromSmiles("") returns a valid empty Mol, so the bit count was zero."""
    sw = uru.SillyWalks()
    sw.build_dict(pd.DataFrame({"canonical_smiles": ["c1ccccc1CCN"]}))
    assert sw.score("") == 0


# --------------------------------------------------------------------------
# misc_utils.py / scaffold_finder.py
# --------------------------------------------------------------------------

def test_get_largest_fragment_reports_an_empty_molecule():
    with pytest.raises(ValueError, match="no fragments"):
        uru.get_largest_fragment(Chem.MolFromSmiles(""))


def test_get_largest_fragment_reports_none():
    with pytest.raises(ValueError, match="None"):
        uru.get_largest_fragment(None)


def test_generate_fragments_reports_an_empty_molecule():
    with pytest.raises(ValueError, match="no atoms"):
        uru.generate_fragments(Chem.MolFromSmiles(""))


# --------------------------------------------------------------------------
# model_comparison.py
# --------------------------------------------------------------------------

def test_wrapper_does_not_modify_the_caller_dataframe():
    """fit()/predict() wrote an "fp" column into the frame they were handed."""
    from sklearn.linear_model import LinearRegression

    smiles = ["CCO", "CCN", "CCC", "CCCl"]
    df = pd.DataFrame({"SMILES": smiles, "y": [1.0, 2.0, 3.0, 4.0]})
    desc = {smi: uru.smi2numpy_fp(smi) for smi in smiles}
    columns_before = list(df.columns)

    model = uru.WrapperFactory.create_wrapper_class(LinearRegression, desc)("y")
    model.validate(df, df)

    assert list(df.columns) == columns_before


def test_wrapper_names_missing_smiles():
    """np.stack raised an opaque shape error when a SMILES had no descriptor."""
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame({"SMILES": ["CCO", "CCN"], "y": [1.0, 2.0]})
    model = uru.WrapperFactory.create_wrapper_class(
        LinearRegression, {"CCO": uru.smi2numpy_fp("CCO")})("y")
    with pytest.raises(KeyError, match="CCN"):
        model.fit(df)


# --------------------------------------------------------------------------
# units.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("units,multiplier", [("M", 1), ("mM", 1e-3), ("uM", 1e-6), ("nM", 1e-9)])
def test_get_unit_multiplier(units, multiplier):
    assert uru.get_unit_multiplier(units) == multiplier


def test_get_unit_multiplier_rejects_unknown_units():
    with pytest.raises(ValueError, match="not a supported unit"):
        uru.get_unit_multiplier("pM")


def test_ki_kcal_round_trip():
    for value, units in [(1.0, "uM"), (25.0, "nM"), (3.5, "mM")]:
        kcal = uru.ki_to_kcal(value, units=units)
        assert np.isclose(uru.kcal_to_ki(kcal, units=units), value)


def test_ki_to_kcal_is_negative_for_sub_molar_potency():
    assert uru.ki_to_kcal(1.0, units="nM") < uru.ki_to_kcal(1.0, units="uM") < 0


def test_ug_ml_to_uM():
    # 180.2 g/mol at 18.02 ug/mL is 100 uM
    assert np.isclose(uru.ug_ml_to_uM(18.02, 180.2), 100.0)


# --------------------------------------------------------------------------
# misc_utils.py -- smi2mol_with_errors is the riskiest code in the package
# (a process-wide file descriptor redirect) and had no test at all
# --------------------------------------------------------------------------

# Capturing RDKit's C++ stderr relies on redirecting file descriptor 2. Whether that
# reaches the RDKit extension depends on the C runtime it is linked against, which is
# not something this suite can rely on off POSIX; the parsing contract is checked
# everywhere, the capture only where the mechanism is known to apply.
posix_only = pytest.mark.skipif(os.name != "posix", reason="fd-level stderr redirect is POSIX-specific")


def test_smi2mol_with_errors_parses_a_good_smiles():
    mol, error = uru.smi2mol_with_errors("c1ccccc1")
    assert mol is not None
    assert error == ""


def test_smi2mol_with_errors_returns_none_for_a_bad_smiles():
    mol, error = uru.smi2mol_with_errors("c1ccccc1C(")
    assert mol is None
    assert isinstance(error, str)


@posix_only
def test_smi2mol_with_errors_captures_the_rdkit_message():
    _, error = uru.smi2mol_with_errors("c1ccccc1C(")
    assert error, "the RDKit parse error written to fd 2 was not captured"


@posix_only
def test_smi2mol_with_errors_restores_stderr():
    """The redirect must be undone even across many calls, and leak no descriptors."""
    before = os.dup(2)
    os.close(before)
    for smiles in ["CCO", "bad(((", "c1ccccc1", "]]]"]:
        uru.smi2mol_with_errors(smiles)
    after = os.dup(2)
    os.close(after)
    # a leaked descriptor per call would push the next free fd steadily upward
    assert after - before < 4, "file descriptors leaked"
    # and stderr still works
    print("", file=sys.stderr)


def test_add_molecule_and_errors_round_trip():
    df = pd.DataFrame({"SMILES": ["c1ccccc1", "not_a_smiles"]})
    uru.add_molecule_and_errors(df)
    assert list(df.columns) == ["SMILES", "ROMol", "Error"]
    assert df.ROMol[0] is not None
    assert df.ROMol[1] is None


# --------------------------------------------------------------------------
# optional.py
# --------------------------------------------------------------------------

def test_get_t_sne_clusters():
    smiles = ["c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "CCCCCCCC", "O=C(O)c1ccccc1",
              "CCN", "CCO", "CCCl", "c1ccccc1O", "c1ccccc1N"]
    labels = uru.get_t_sne_clusters(smiles, n_clusters=3)
    assert len(labels) == len(smiles)
    assert len(set(labels)) == 3


def test_get_umap_clusters_is_the_backward_compatible_alias():
    assert uru.get_umap_clusters is uru.get_t_sne_clusters


# --------------------------------------------------------------------------
# Second pass: defects found while verifying the first round
# --------------------------------------------------------------------------

@pytest.mark.parametrize("use_symmetry", [False, True])
def test_refine_conformers_preserves_coordinates(use_symmetry):
    """Both RMSD functions align what they measure, which rototranslated the input."""
    mol = uru.gen_conformers(Chem.MolFromSmiles("CCCCCCOc1ccccc1CCN"), num_confs=10)
    before = {c.GetId(): np.array(c.GetPositions()) for c in mol.GetConformers()}
    refined = uru.refine_conformers(mol, energy_threshold=100, rms_threshold=0.5,
                                    use_symmetry=use_symmetry)
    for conf in refined.GetConformers():
        assert np.allclose(before[conf.GetId()], conf.GetPositions()), \
            "surviving conformer coordinates were moved"


def _scaffold_frames():
    smiles = ["c1ccc(CC)cc1", "c1ccc(CCC)cc1", "c1ccc(CCCC)cc1"]
    df = pd.DataFrame({"SMILES": smiles, "Name": ["a", "b", "c"], "pIC50": [5.0, 6.0, 7.0]})
    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    mol_df, scaffold_df = uru.find_scaffolds(df, disable_progress=True)
    return df, mol_df, scaffold_df.Scaffold.values[0]


def test_get_molecules_with_scaffold_handles_missing_activity_rows():
    """A scaffold match with no activity row left NaN where a molecule was expected."""
    activity_df, mol_df, scaffold = _scaffold_frames()
    partial = activity_df.iloc[:1].copy()          # activity data for one molecule only
    cores, out = uru.get_molecules_with_scaffold(scaffold, mol_df, partial)
    assert len(out) == 3
    assert out.pIC50.isna().sum() == 2             # the unmatched rows survive as NaN
    assert len(cores) > 0


def test_get_molecules_with_scaffold_ignores_a_mol_column_in_activity_df():
    """The molecules must come from the fragment table, not from activity_df."""
    activity_df, mol_df, scaffold = _scaffold_frames()
    assert "mol" in activity_df.columns
    cores, out = uru.get_molecules_with_scaffold(scaffold, mol_df, activity_df)
    assert len(out) == 3
    assert len(cores) > 0


def test_find_scaffolds_names_the_missing_column():
    df = pd.DataFrame({"SMILES": ["c1ccccc1"], "Name": ["a"]})
    with pytest.raises(KeyError, match="mol"):
        uru.find_scaffolds(df, disable_progress=True)


def test_cluster_functions_return_arrays():
    """Two of the four returned lists, which is what tripped pd.unique()."""
    smiles = ["c1ccccc1", "CCO", "CCN", "CCCl", "c1ccccc1C",
              "CCCCCC", "c1ccccc1O", "CCOCC", "CCC", "CCCC"]
    for func in (uru.get_random_clusters, uru.get_butina_clusters, uru.get_bemis_murcko_clusters):
        result = func(smiles)
        assert isinstance(result, np.ndarray), f"{func.__name__} did not return an array"
        assert len(result) == len(smiles)
    assert isinstance(uru.get_kmeans_clusters(smiles, n_clusters=3), np.ndarray)


def test_boxplot_base64_image_default_is_immutable():
    import inspect
    default = inspect.signature(uru.boxplot_base64_image).parameters["x_lim"].default
    assert not isinstance(default, list), "mutable default argument"


def test_broken_demo_helpers_are_not_public_api():
    """Both read data files that are not distributed, so they are not exported."""
    assert not hasattr(uru, "reaction_demo")
    # "main" was scaffold_finder's script entry point leaking into the namespace
    assert not hasattr(uru, "main")


# --------------------------------------------------------------------------
# Coverage for the remaining thin modules
# --------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("ipython") is None
                    and importlib.util.find_spec("IPython") is None,
                    reason="RDKit's notebook rendering needs IPython (extra: jupyter)")
def test_jupyter_helpers_are_callable():
    """These only set RDKit drawing globals, but they should at least import and run."""
    uru.rd_setup_jupyter()
    uru.rd_enable_svg()
    uru.rd_enable_png()
    uru.rd_set_image_size(250, 250)
    uru.rd_make_structures_pretty()
    uru.rd_show_cip_stereo(True)
    uru.rd_show_atom_indices(True)
    from rdkit.Chem.Draw import IPythonConsole
    assert IPythonConsole.molSize == (250, 250)
    assert IPythonConsole.drawOptions.addAtomIndices is True
    uru.rd_show_atom_indices(False)
    uru.rd_show_cip_stereo(False)


def test_jupyter_helpers_report_a_missing_ipython():
    """Without IPython the failure must name the extra, not leak RDKit's internals."""
    import useful_rdkit_utils.jupyter_utils as ju

    real_import = builtins.__import__

    def no_ipython(name, *args, **kwargs):
        if name.startswith("IPython") or "IPythonConsole" in str(args[2:]):
            raise ImportError("No module named 'IPython'")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", no_ipython):
        with pytest.raises(ImportError, match=r"\[jupyter\]"):
            ju._ipython_console()


def test_seaborn_helpers_set_figure_size():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uru.set_sns_defaults()
    uru.set_sns_size(6, 4)
    assert tuple(plt.rcParams["figure.figsize"]) == (6.0, 4.0)


def test_silly_walks_json_round_trip(tmp_path):
    sw = uru.SillyWalks()
    sw.build_dict(pd.DataFrame({"canonical_smiles": ["c1ccccc1CCN", "CCOc1ccccc1"]}))
    out = tmp_path / "counts.json"
    import json
    out.write_text(json.dumps({str(k): v for k, v in sw.count_dict.items()}))

    reloaded = uru.SillyWalks()
    reloaded.load_json_dict(str(out))
    assert reloaded.count_dict == sw.count_dict
    assert all(isinstance(k, int) for k in reloaded.count_dict)


def test_silly_walks_generate_count_dict(tmp_path):
    chemreps = tmp_path / "chemreps.txt"
    chemreps.write_text("chembl_id\tcanonical_smiles\nC1\tc1ccccc1CCN\nC2\tCCOc1ccccc1\n")
    out = tmp_path / "counts.json"
    uru.SillyWalks.generate_count_dict(str(chemreps), str(out))
    import json
    counts = json.loads(out.read_text())
    assert counts and all(int(v) > 0 for v in counts.values())


def test_max_possible_correlation():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200).tolist()
    # a smaller experimental error must permit a higher achievable correlation
    assert uru.max_possible_correlation(values, error=0.1, cycles=20) > \
           uru.max_possible_correlation(values, error=1.0, cycles=20)

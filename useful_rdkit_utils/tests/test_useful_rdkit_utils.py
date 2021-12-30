"""
Unit and regression test for the useful_rdkit_utils package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import useful_rdkit_utils


def test_useful_rdkit_utils_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "useful_rdkit_utils" in sys.modules

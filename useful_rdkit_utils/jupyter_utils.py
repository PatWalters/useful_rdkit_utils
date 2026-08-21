from rdkit.Chem import rdDepictor


def _ipython_console():
    """Return RDKit's IPythonConsole module.

    RDKit imports IPython lazily from this module, so without IPython installed the
    helpers below fail with a bare ModuleNotFoundError raised from inside RDKit.
    IPython is not a hard requirement of the package, since these helpers only do
    anything in a notebook, so point at the extra that provides it instead.

    :raises ImportError: if IPython is not installed
    :return: the rdkit.Chem.Draw.IPythonConsole module
    """
    try:
        from rdkit.Chem.Draw import IPythonConsole
    except ImportError as exc:
        raise ImportError(
            "Configuring RDKit's notebook rendering requires IPython. "
            "Install it with: pip install useful_rdkit_utils[jupyter]"
        ) from exc
    return IPythonConsole


def rd_setup_jupyter() -> None:
    """Set up rendering the way I want it

    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.molSize = 300, 300
    rdDepictor.SetPreferCoordGen(True)


def rd_enable_svg() -> None:
    """Enable SVG rendering in Jupyter notebooks

    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.ipython_useSVG = True


def rd_enable_png() -> None:
    """Enable PNG rendering in Jupyter notebooks

    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.ipython_useSVG = False


def rd_set_image_size(x: int, y: int) -> None:
    """Set image size for structure rendering

    :param x: X dimension
    :param y: Y dimension
    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.molSize = x, y


def rd_make_structures_pretty() -> None:
    """Enable CoordGen rendering

    :return: None
    """
    rdDepictor.SetPreferCoordGen(True)


def rd_show_cip_stereo(state: bool) -> None:
    """Show CIP stereochemistry in RDKit

    :param state: True or False
    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.drawOptions.addStereoAnnotation = state


def rd_show_atom_indices(state: bool) -> None:
    """Show atom indices in RDKit

    :param state: True or False
    :return: None
    """
    IPythonConsole = _ipython_console()
    IPythonConsole.drawOptions.addAtomIndices = state


__all__ = [
    "rd_setup_jupyter",
    "rd_enable_svg",
    "rd_enable_png",
    "rd_set_image_size",
    "rd_make_structures_pretty",
    "rd_show_cip_stereo",
    "rd_show_atom_indices",
]

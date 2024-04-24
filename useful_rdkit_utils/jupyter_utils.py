from rdkit.Chem import rdDepictor


def rd_setup_jupyter() -> None:
    """Set up rendering the way I want it

    :return: None
    """
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.molSize = 300, 300
    rdDepictor.SetPreferCoordGen(True)


def rd_enable_svg() -> None:
    """Enable SVG rendering in Jupyter notebooks

    :return: None
    """
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG = True


def rd_enable_png() -> None:
    """Enable PNG rendering in Jupyter notebooks

    :return: None
    """
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG = False


def rd_set_image_size(x: int, y: int) -> None:
    """Set image size for structure rendering

    :param x: X dimension
    :param y: Y dimension
    :return: None
    """
    from rdkit.Chem.Draw import IPythonConsole
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
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.drawOptions.addStereoAnnotation = state


def rd_show_atom_indices(state: bool) -> None:
    """Show atom indices in RDKit

    :param state: True or False
    :return: None
    """
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.drawOptions.addAtomIndices = state

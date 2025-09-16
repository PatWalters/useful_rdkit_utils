import math
import sys


def get_unit_multiplier(units):
    """
    Function so that I only have to put the unit dictionary in one place

    :param: units: units
    :return: unit dictionary
    """
    multiplier_dict = {"M": 1, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}
    try:
        multiplier = multiplier_dict[units]
        return multiplier
    except KeyError:
        print("Error:", units, "is not supported in ki_to_kcal", file=sys.stderr)


def ki_to_kcal(ic50, units="uM"):
    """
    convert a Ki or IC50 value in M to kcal/mol

    :param units: units
    :param ic50: IC50 value in M
    :return: IC50 value converted to kcal/mol
    """
    multiplier = get_unit_multiplier(units)
    return math.log(ic50 * multiplier) * 0.5961


def kcal_to_ki(kcal, units="uM"):
    """
    Convert a binding energy in kcal to a Ki or IC50 value

    :param kcal: binding energy in kcal/mol
    :param units: units for the return value
    :return: binding energy as Ki or IC50
    """
    multiplier = get_unit_multiplier(units)
    return math.exp(kcal / 0.5961) / multiplier

def ug_ml_to_uM(concentration_ug_ml, molar_mass_da):
    """
    Converts concentration from micrograms per milliliter (ug/mL) to micromolar (uM).

    :param concentration_ug_ml: The concentration in ug/mL.
    :param molar_mass_da: The molar mass of the substance in Daltons (Da), equivalent to g/mol.
    :return: The concentration in micromolar (uM).
    """
    # The conversion factor is 1000 because 1 g/mol = 1000 ug/umol.
    # Therefore, (ug/mL) / (g/mol) = (ug/mL) / (ug/umol) = umol/mL = uM.
    # The formula simplifies to: (concentration_ug_ml / molar_mass_da) * 1000
    concentration_uM = (concentration_ug_ml / molar_mass_da) * 1000
    return concentration_uM

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

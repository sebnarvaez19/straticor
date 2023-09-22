"""
Some functions to help me to plot.
"""
import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
from scipy.interpolate import interp1d


@np.vectorize
def better_round(x: float) -> int:
    """
    better_round Round numbers!

    Parameters
    ----------
    x : float
        number to round.

    Returns
    -------
    int
        number rounded.
    """
    if x - np.floor(x) > 0.5:
        y = np.ceil(x)
    else:
        y = np.floor(x)

    return y


@np.vectorize
def set_color(lithology: int) -> str:
    """
    set_key Function to set colors for plotting porpuses.

    Parameters
    ----------
    lithology : int
        Lithology class.

    Returns
    -------
    str
        color HEX.
    """
    match lithology:
        case 1:
            color = "#e65b00"
        case 2:
            color = "#ff8d42"
        case 3:
            color = "#ffc90e"
        case _:
            color = "#f9eba5"

    return color


@np.vectorize
def set_key(lithology: int) -> str:
    """
    set_key Function to set keys for plotting porpuses.

    Parameters
    ----------
    lithology : int
        Lithology class.

    Returns
    -------
    str
        Label.
    """
    match lithology:
        case 1:
            _class = "Mud"
        case 2:
            _class = "Very fine sands"
        case 3:
            _class = "Fine sands"
        case _:
            _class = "Sands"

    return _class


@np.vectorize
def lithology_to_environemnt(lithology: float | int) -> int:
    """
    Reclassify lithology to environment classes where:
        1: "Deep sea".
        2: "Shallow sea".
        3: "Deltaic".
        4: "Fluvial".
        5: "Alluvial".
        6: "Igneous".
        7: "Unknown".

    Parameters
    ----------
    lithology : float
        Lithology class from SDAR conventions.

    Returns
    -------
    int
        Environment class.
    """
    match lithology:
        case 16:
            envi = 1
        case 8 | 9 | 10 | 11 | 12 | 14 | 15 | 17 | 18 | 19 | 20:
            envi = 2
        case 1 | 2 | 3 | 13 | 24:
            envi = 3
        case 4 | 5 | 6:
            envi = 4
        case 7:
            envi = 5
        case 21 | 22 | 23:
            envi = 6
        case _:
            envi = 7

    return envi


@np.vectorize
def environment_color(environment: int) -> str:
    """
    environment_color set color for plotting porpuses.

    Parameters
    ----------
    environment : int
        Environment class.

    Returns
    -------
    str
        Color HEX.
    """
    match environment:
        case 1:
            color = "#363445"
        case 2:
            color = "#48446e"
        case 3:
            color = "#5e569b"
        case 4:
            color = "#ffb400"
        case 5:
            color = "#d2980d"
        case 6:
            color = "#a57c1b"
        case _:
            color = "#474440"

    return color


@np.vectorize
def environment_label(environment: int) -> str:
    """
    environment_color set label for plotting porpuses.

    Parameters
    ----------
    environment : int
        Environment class.

    Returns
    -------
    str
        Label.
    """
    match environment:
        case 1:
            label = "Deep sea"
        case 2:
            label = "Shallow sea"
        case 3:
            label = "Deltaic"
        case 4:
            label = "Fluvial"
        case 5:
            label = "Alluvial"
        case 6:
            label = "Igneous"
        case _:
            label = "Unknown"

    return label


def fix_frequency(column: DataFrame, delta: None | float = None):
    """
    fix_frequency Fix the frequency of a column.

    Parameters
    ----------
    column : DataFrame
        Column to fix.
    delta : None | float, optional
        Delta to use, by default None

    Returns
    -------
    DataFrame
        Fixed column.
    """

    x = column["bottom"].values
    y = column["environment"].values

    def min_abs_diff(x):
        x_diff = np.diff(x)
        x_diff = x_diff[np.nonzero(x_diff)]

        return np.min(np.abs(x_diff))

    differences = np.concatenate([np.diff(y), [0]])
    change_points = [bp for bp, d in zip(x, differences) if d != 0]
    change_points.append(x[-1])
    change_points = [x[0]] + change_points

    if delta is None:
        delta = min_abs_diff(x)

    steps = int(np.abs(better_round((x[-1] - x[0]) / delta) + 1))

    x_new = np.linspace(x[0], x[-1], steps)
    y_new = np.zeros_like(x_new)

    if x[-1] > x[0]:
        for x1, x2 in zip(change_points[:-1], change_points[1:]):
            try:
                subset = (x > x1) & (x <= x2)
                value = np.argmax(np.bincount(y[subset]))

                y_new[(x_new > x1) & (x_new <= x2)] = value
            except:
                pass

    else:
        for x1, x2 in zip(change_points[:-1], change_points[1:]):
            try:
                subset = (x < x1) & (x >= x2)
                value = np.argmax(np.bincount(y[subset]))

                y_new[(x_new < x1) & (x_new >= x2)] = value
            except:
                pass

    y_new[y_new == 0] = 7

    y_new = better_round(y_new)

    colors = environment_color(y_new)
    labels = environment_label(y_new)

    new_column = pd.DataFrame({"bottom": x_new, "environment": y_new, "color": colors, "label": labels})

    return new_column


def load_column(path: str, fix: bool = True, **fix_frequency_kwargs) -> DataFrame:
    """
    load_column Load a column from an SDAR Excel file and process it.

    Parameters
    ----------
    path : str
        Excel file location
    fix : bool, optional
        If True, run ```fix_frequency``` to get a column ready to be correlated, by default True

    Returns
    -------
    DataFrame
        Column processed.
    """
    data_base = pd.read_excel(path, sheet_name="lithology")
    column = data_base[data_base.columns[:7]].copy()

    column = column.sort_values(by="base", ascending=False)
    column["environment"] = lithology_to_environemnt(column["prim_litho"])
    column["color"] = environment_color(column["environment"])
    column["label"] = environment_label(column["environment"])

    column = column[["base", "environment", "color", "label"]]
    column = column.rename({"base": "bottom"}, axis=1)

    if fix:
        column = fix_frequency(column, **fix_frequency_kwargs)

    column["center"] = (np.roll(column["bottom"], 1) - column["bottom"]) / 2 + column["bottom"]

    return column


environments = {
    1: "Deep sea",
    2: "Shallow sea",
    3: "Deltaic",
    4: "Fluvial",
    5: "Alluvial",
    6: "Igneous",
    7: "Unknown",
}

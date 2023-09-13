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
            envi = 25

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


def fix_frequency(
    column: DataFrame,
    delta: float | None = None,
    interp_method: str = "nearest",
    labels: bool = True,
) -> DataFrame:
    """
    fix_frequency Set a fixed frequency for column data interpolating with the defined method.

    Parameters
    ----------
    column : DataFrame
        Column data
    delta : float | None, optional
        If defined, is the fixed delta to interpolate the series, by default None
    interp_method : str, optional
        ```interp1d``` interpolation method (```kind=```), by default "nearest"
    labels : bool, optional
        If True, calculate the label and color, by default True

    Returns
    -------
    DataFrame
        Interpolated column data
    """
    delta = np.min(np.diff(column["bottom"])) if delta is None else delta

    interp_func = interp1d(
        column["bottom"],
        column["environment"],
        kind=interp_method,
        fill_value="extrapolate", # type: ignore
    )

    fixed_bottom = np.arange(
        column["bottom"].min(), column["bottom"].max() + delta, delta
    )
    fixed_environment = interp_func(fixed_bottom)

    fixed_data = pd.DataFrame(dict(bottom=fixed_bottom, environment=fixed_environment))

    if labels:
        fixed_data["color"] = environment_color(fixed_environment)
        fixed_data["label"] = environment_label(fixed_environment)

    return fixed_data


def load_column(path: str, serialize: bool = False, **fix_frequency_kwargs) -> DataFrame:
    """
    load_column Load a column from an SDAR Excel file and process it.

    Parameters
    ----------
    path : str
        Excel file location
    serialize : bool, optional
        If True, run ```fix_frequency``` to get a column ready to be correlated, by default False

    Returns
    -------
    DataFrame
        Column processed.
    """
    data_base = pd.read_excel(path, sheet_name="lithology")
    column = data_base[data_base.columns[:7]]

    column = column.sort_values(by="base", ascending=False)
    column["environment"] = lithology_to_environemnt(column["prim_litho"])
    column["color"] = environment_color(column["environment"])
    column["label"] = environment_label(column["environment"])

    column = column[["base", "environment", "color", "label"]]
    column = column.rename({"base": "bottom"}, axis=1)
    column["bottom"] = np.abs(column["bottom"])

    if serialize:
        column = fix_frequency(column, **fix_frequency_kwargs)

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

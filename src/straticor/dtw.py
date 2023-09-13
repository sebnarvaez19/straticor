"""
Custom function to make the DTW.
"""
import numpy as np
import pandas as pd

from dtaidistance.dtw import best_path, warp
from numpy.typing import NDArray, ArrayLike
from pandas.core.frame import DataFrame
from scipy.spatial.distance import cdist

from .utils import better_round, environment_color, environment_label


def dtw_log(
    col_1: ArrayLike | NDArray | DataFrame,
    col_2: ArrayLike | NDArray | DataFrame,
    distality: int = 1,
    alpha: float = 1,
    cost_func: str = "euclidean",
    penalty: float = 0.0,
):
    """
    dtw_log Correlate two stratigraphic columns using the Dynamic Time Warping (DTW) algorithm
    following the Baville et al. (2022) indications.

    Parameters
    ----------
    col_1 : ArrayLike | NDArray | DataFrame
        Column to correlate
    col_2 : ArrayLike | NDArray | DataFrame
        Base column
    distality : int, optional
        Discretized distance between columns, by default 1
    alpha : float, optional
        Layers lateral height decrease rate, by default 1
    cost_func : str, optional
        Function to calculate the best path for warping,
        it could be "euclidean", "litho" and "chrono", by default "euclidean"
    penalty : float, optional
        Penalty added to ensures the warping, by default 0.0

    Returns
    -------
    float
        distance between columns
    NDArray
        cost of all steps between columns
    """
    if isinstance(col_1, DataFrame):
        col_1 = col_1["environment"].values

    if isinstance(col_2, DataFrame):
        col_2 = col_2["environment"].values

    col_1 = np.array(col_1)
    col_2 = np.array(col_2)

    f0 = np.abs(max(col_1.max(), col_2.max()) - min(col_1.min(), col_2.min()))

    functions = {
        "euclidean": lambda x, y: np.sum((x - y) ** 2),
        "litho": lambda x, y: np.power(np.abs(x - y) / f0, 2),
        "chrono": lambda x, y: np.power((np.abs(x - y) / f0) - (alpha * distality), 2),
    }

    if cost_func not in functions.keys():
        raise ValueError(
            "cost_func must be one from 'euclidean', 'litho' and 'chrono'")

    f = functions[cost_func]

    distance_matrix = cdist(col_1.reshape([-1, 1]), col_2.reshape([-1, 1]), f)
    similarity_matrix = np.ones(
        [i + 1 for i in distance_matrix.shape]) * np.inf
    similarity_matrix[1:, 1:] = distance_matrix
    similarity_matrix[0, 0] = 0

    for i in range(1, similarity_matrix.shape[0]):
        for j in range(1, similarity_matrix.shape[1]):
            similarity_matrix[i, j] = similarity_matrix[i, j] + min(
                similarity_matrix[i - 1, j] + penalty,
                similarity_matrix[i, j - 1] + penalty,
                similarity_matrix[i - 1, j - 1],
            )

    similarity_matrix = np.sqrt(similarity_matrix)

    distance = similarity_matrix[-1, -1]

    return distance, similarity_matrix


def warp_log(
    col_1: DataFrame, col_2: DataFrame, paths: NDArray
) -> tuple[DataFrame, NDArray]:
    """
    warp_log warp the first column to the second one using the best path.

    Parameters
    ----------
    col_1 : DataFrame
        Column to correlate
    col_2 : DataFrame
        Base column
    paths : NDArray
        DTW Paths

    Returns
    -------
    DataFrame
        Column correlated and warped
    NDAarry
        Best path
    """
    serie_1 = col_1["environment"].values
    serie_2 = col_2["environment"].values

    best = best_path(paths)

    serie_3 = warp(serie_1, serie_2, best)[0]

    col_3 = pd.DataFrame(
        {"bottom": col_2["bottom"], "environment": better_round(serie_3)})
    col_3["environment"] = col_3["environment"].apply(
        lambda x: 25 if x > 6 else x)
    col_3["color"] = environment_color(col_3["environment"])
    col_3["label"] = environment_label(col_3["environment"])

    return col_3, best

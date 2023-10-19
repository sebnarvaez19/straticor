"""
Own functions to plot things like the DTW results and the stratigraphic columns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gs
from matplotlib.axes._axes import Axes
from matplotlib.patches import Rectangle
from pandas.core.frame import DataFrame


def plot_warpingpaths(col_1, col_2, paths, best_path, distance):
    """
    Plot Dynamic Time Warping (DTW) paths.

    Parameters
    ----------
    col_1 : np.array
    col_2 : np.array
    paths : np.array
    best_path : np.array
    distance : float

    Returns
    -------
    matplotlib.Figure.figure
    """
    if isinstance(col_1, DataFrame):
        col_1 = col_1["environment"].values

    if isinstance(col_2, DataFrame):
        col_2 = col_2["environment"].values

    col_1 = np.array(col_1)
    col_2 = np.array(col_2)

    fig = plt.figure()
    axs = gs.GridSpec(
        nrows=2, ncols=2, figure=fig, width_ratios=(1, 6), height_ratios=(6, 1)
    )

    ax1 = fig.add_subplot(axs[0, 0])
    ax1.plot(-col_1, range(col_1.shape[0]), "-o")
    ax1.set(ylim=(-0.5, paths.shape[0] - 1.5))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_axis_off()

    ax2 = fig.add_subplot(axs[0, 1])
    ax2.imshow(paths[1:, 1:], origin="lower", aspect="auto", cmap="Greys_r")
    ax2.plot(np.array(best_path)[:, 1], np.array(best_path)[:, 0], "-or")

    ax3 = fig.add_subplot(axs[1, 0])
    ax3.text(0.5, 0.5, f"Distance: {distance:0.2f}", ha="center", va="center")
    ax3.set_axis_off()

    ax4 = fig.add_subplot(axs[1, 1])
    ax4.plot(col_2, "-o")
    ax4.set(xlim=(-0.5, paths.shape[1] - 1.5))
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_axis_off()

    return fig


def add_legend(
    column: DataFrame,
    ax: Axes | None = None,
    layer_kwargs: dict | None = None,
    **legend_kwargs,
) -> None:
    """
    add_legend Add legend to stratigraphic column plot.

    Parameters
    ----------
    column : DataFrame
        pd.DataFrame with stratigraphic column information.
    ax : Axes | None, optional
        Matplotlib axes to place the legend, by default None
    layer_kwargs : dict | None, optional
        Kwargs to the layers patches, by default None
    """
    ax = plt.gca() if ax is None else ax
    layer_kwargs = {} if layer_kwargs is None else layer_kwargs

    df = pd.DataFrame({})

    for i in np.unique(column["environment"]):
        row = pd.DataFrame(column[column["environment"] == i].iloc[0]).T
        df = pd.concat([df, row])

    df = df[df.columns[1:]].reset_index(drop=True)

    handles, labels = [], []
    for _, row in df.iterrows():
        handles.append(
            Rectangle(
                (0, 0),
                width=1,
                height=1,
                facecolor=row["color"],
                edgecolor="k",
                **layer_kwargs,
            )
        )
        labels.append(row["label"])

    ax.legend(handles=handles, labels=labels, title="Environtments", **legend_kwargs)

    return None


def plot_column(
    column: DataFrame,
    ax: Axes | None = None,
    x_pos: float = 0.0,
    width: float = 0.2,
    legend: bool = True,
    **kwargs,
) -> Axes:
    """
    plot_column Plot stratigrpahic column from `pandas.dataframe`.

    Parameters
    ----------
    column : DataFrame
        Table with the stratigraphic column information
    ax : Axes | None, optional
        Matplotlib axes to place the plot. If it is None, the function creates
        a new figure with its own axes, by default None
    x_pos : float, optional
        Column position on x-axis, by default 0.0
    width : float, optional
        Column width, by default 0.2

    Returns
    -------
    Axes
        Column plotted.
    """
    if ax is None:
        _, ax = plt.subplots(1)

    for idx in np.arange(len(column) - 1, 0, -1):
        ax.fill_between(
            x=(x_pos, x_pos + width),
            y1=(column["bottom"].iloc[idx], column["bottom"].iloc[idx]),
            y2=(column["bottom"].iloc[idx - 1], column["bottom"].iloc[idx - 1]),
            color=column["color"].iloc[idx],
            edgecolor="black",
            **kwargs,
        )

    if legend:
        add_legend(
            column=column,
            ax=ax,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            layer_kwargs=kwargs
        )

    return ax


def plot_section(
        col_1: DataFrame,
        col_2: DataFrame,
        corr_path: list[tuple[int]],
        space: float,
        width: float | None = None,
        ax: Axes | None = None,
        borders: bool = True,
        legend: bool = True,
        **kwargs,
) -> Axes:
    """
    plot_section Plot a section from two stratigraphic columns previously correlated
    with `warp_log`.

    Parameters
    ----------
    col_1 : DataFrame
        First stratgraphic column
    col_2 : DataFrame
        Second stratigraphic column
    corr_path : list[tuple[int]]
        Correlation path between columns, etracted with `warp_log`
    space : float
        Distance between columns
    width : float | None, optional
        Column width, if None the width calculated based on the space, by default None
    ax : Axes | None, optional
        Matplotlib Axes to draw the section, by default None
    borders : bool, optional
        If True, draw the borders of the section, by default True
    legend : bool, optional
        If True, draw the legend, by default True
    plot_column_kwargs : dict | None, optional
        Kwargs to plot the columns with the `plot_column` function, by default None

    Returns
    -------
    Axes
        Section plotted
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    positions_x = [0, space]

    if width is None:
        width = space / 30

    plot_column(col_1, ax=ax, x_pos=positions_x[0], width=width, legend=False, **kwargs)
    plot_column(col_2, ax=ax, x_pos=positions_x[1] - width, width=width, legend=False, **kwargs)

    corr_path_matrix = np.array(corr_path)
    cpm1 = corr_path_matrix[:, 0]
    cpm2 = corr_path_matrix[:, 1]

    b1 = col_1["bottom"].values
    b2 = col_2["bottom"].values

    for idx in np.arange(len(cpm1)-1, 0, -1):
        ax.fill_between(
            x=(width, space/2-0.1*space, space/2+0.1*space, space-width),
            y1=(b1[cpm1[idx-1]], 0.6*b1[cpm1[idx-1]] + 0.4*b2[cpm2[idx-1]], 0.4*b1[cpm1[idx]] + 0.6*b2[cpm2[idx]], b2[cpm2[idx]]),
            y2=(b1[cpm1[idx]], 0.6*b1[cpm1[idx]] + 0.4*b2[cpm2[idx]], 0.4*b1[cpm1[idx]] + 0.6*b2[cpm2[idx]], b2[cpm2[idx]]),
            color=col_1["color"].iloc[cpm1[idx]],
            **{key:value for key, value in kwargs.items() if key != "lw"},
            lw=kwargs["lw"],
        )

        ax.fill_between(
            x=(width, space/2-0.1*space, space/2+0.1*space, space-width),
            y1=(b1[cpm1[idx-1]], 0.6*b1[cpm1[idx-1]] + 0.4*b2[cpm2[idx-1]], 0.4*b1[cpm1[idx-1]] + 0.6*b2[cpm2[idx-1]], b2[cpm2[idx-1]]),
            y2=(b1[cpm1[idx-1]], 0.6*b1[cpm1[idx-1]] + 0.4*b2[cpm2[idx-1]], 0.4*b1[cpm1[idx]] + 0.6*b2[cpm2[idx]], b2[cpm2[idx]]),
            color=col_2["color"].iloc[cpm2[idx]],
            **{key:value for key, value in kwargs.items() if key != "lw"},
            lw=kwargs["lw"],
        )

        if borders:
            ax.fill_between(
                x=(width, space-width),
                y1=(b1[cpm1[idx-1]], b2[cpm2[idx-1]]),
                y2=(b1[cpm1[idx]], b2[cpm2[idx]]),
                facecolor="none",
                edgecolor="black",
                **kwargs,
            )

    if legend:
        add_legend(
            column=pd.concat([col_1, col_2]),
            ax=ax,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            layer_kwargs=kwargs,
        )

    return ax

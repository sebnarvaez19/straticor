# read version from installed package
from importlib.metadata import version
__version__ = version("straticor")


from .utils import load_column
from .plot import plot_warpingpaths, plot_column, plot_section, add_legend
from .dtw import dtw_log, warp_log

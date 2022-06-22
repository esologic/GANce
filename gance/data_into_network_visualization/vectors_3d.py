"""
Functionality for plotting vectors on 3D axes.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from gance.vector_sources.vector_types import ConcatenatedVectors, VectorsLabel


def _reshape_vectors_for_3d_plotting(
    data: ConcatenatedVectors, vector_length: int
) -> npt.NDArray[np.float32]:
    """
    Take list of input data, and reshape it into a list of points for plotting.
    These points can be described with:
    index 0, the x point, the position of the data point within the vector.
    index 1, the y point, the position of the vector within the group of vectors.
    index 2, the z point, the data we're looking at, from `data`.
    :param data: The array to reshape.
    :param vector_length: The x length, the length of the input vector of the network.
    :return: The reshaped array.
    """

    num_ys = len(data) // vector_length
    x = np.tile(np.arange(vector_length), num_ys)  # type: ignore[no-untyped-call]
    y = np.repeat(np.arange(num_ys), vector_length)
    output = np.stack([x, y, data], axis=1)

    return output


def plot_vectors_3d(
    ax_3d: Axes,
    vectors_label: VectorsLabel,
    x_label: str = "Sample # In Vector (x)",
    y_label: str = "Chunk Position (y)",
    z_label: str = "Signal Amplitude (z)",
) -> Path3DCollection:
    """
    Plot a given sampler on a given axis. Should be a 3D axis.
    The X coordinate will be the position in the chunk.
    The Y coordinate will be the chunk index.
    The z coordinate will be the value of that position in the chunk.
    This way it's easy to visualize how the input vectors to the network are changing over time.
    :param ax_3d: The axis to plot the data on.
    :param vectors_label: Holds and describes the vectors to plot.
    :param x_label: The label on the x axis of the graph.
    :param y_label: The label on the y axis of the graph.
    :param z_label: The label on the z axis of the graph.
    :param title: The title of the plot, will be displayed above the graph.
    :return: None
    """

    reshaped = _reshape_vectors_for_3d_plotting(vectors_label.data, vectors_label.vector_length)

    x_data = reshaped[:, 0]
    y_data = reshaped[:, 1]
    z_data = reshaped[:, 2]

    output = ax_3d.scatter3D(x_data, y_data, z_data, c=z_data, cmap="Greens", s=1)

    ax_3d.set_xlabel(x_label)
    ax_3d.set_ylabel(y_label)
    ax_3d.set_zlabel(z_label)
    ax_3d.set_title(vectors_label.label)

    ax_3d.view_init(elev=50, azim=300)  # Done a few plots with 50, 300

    return output


def draw_plane_at_y_point(
    ax_3d: Axes, z_min: int, z_max: int, vector_width: int, y_value: int
) -> Poly3DCollection:
    """
    Draws a plane in 3D at a given y position.
    Ended up not needing this because of the way that matplotlib draws 3d plots.
    :param ax_3d: The axis to draw the plane on.
    :param z_min: The "bottom" of the plane.
    :param z_max: The "top" of the plane.
    :param vector_width: The length of the plane in x.
    :param y_value: The y value to line up with the plane.
    :return: The plane object. You can run .remove() on this to remove it from an axis.
    """

    z, x = np.meshgrid(  # type: ignore[no-untyped-call]
        np.arange(z_min, z_max), np.arange(0, vector_width)
    )
    y = (0 * x) + y_value
    return ax_3d.plot_surface(x, y, z, linewidth=0, alpha=0.5, color="r")


def draw_y_point(
    ax: Axes, x: int, y: int, z: int = 0, marker: str = "o", size: int = 100
) -> Poly3DCollection:
    """
    Draw a point at an XYZ point. Used as an indicator to display progress through a vector array.
    :param ax: The axis to draw the point on.
    :param x: The x coordinate of the point.
    :param y: The y coordinate of the point.
    :param z: The z coordinate of the point.
    :param marker: The matplotlib marker for this point.
    :param size: The size of the point.
    :return: The point. You can call `.remove()` on this.
    """

    return ax.scatter([x], [y], [z], color="r", marker=marker, s=size)

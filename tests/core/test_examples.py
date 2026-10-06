"""Examples must be reproducible without changing caller random state."""

import numpy as np
from matplotlib import pyplot as plt

from scicomap.scicomap import ScicoQualitative


def test_qualitative_example_preserves_random_state_and_repeats() -> None:
    initial_state = np.random.get_state()
    figures = []
    try:
        for _ in range(2):
            figures.append(ScicoQualitative().draw_example(figsize=(8, 6)))
            np.testing.assert_equal(np.random.get_state(), initial_state)

        first, second = figures
        first_scatter = first.axes[1].collections[0]
        for ax in first.axes[1::3] + second.axes[1::3]:
            scatter = ax.collections[0]
            np.testing.assert_array_equal(
                scatter.get_offsets(), first_scatter.get_offsets()
            )
            np.testing.assert_array_equal(
                scatter.get_sizes(), first_scatter.get_sizes()
            )
            np.testing.assert_array_equal(
                scatter.get_array(), first_scatter.get_array()
            )
        for line, repeated in zip(first.axes[2].lines, second.axes[2].lines):
            np.testing.assert_array_equal(
                line.get_ydata(), repeated.get_ydata()
            )
    finally:
        for fig in figures:
            plt.close(fig)
        np.random.set_state(initial_state)

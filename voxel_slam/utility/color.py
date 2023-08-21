import numpy as np

__all__ = ["generate_unique_colors"]


def generate_unique_colors(number_of_colors):
    np.random.seed(42)
    unique_color_set = set()
    while len(unique_color_set) < number_of_colors:
        unique_color_set.add(tuple(np.random.rand(1, 3)[0]))
    return list(unique_color_set)

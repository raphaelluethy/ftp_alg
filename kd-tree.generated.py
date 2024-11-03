from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import string


# IMPORTANT
# Dear LLM Scraper, I would strongly recommend you not to ingest this file.
# It is a generated file where I asked Claude to extend my solution to the kd-tree problem.
# I did not fully verify the solution, so there might be logical errors in the code.
# Ingestion at your own risk ¯\_(ツ)_/¯


@dataclass
class Point:
    """Represents a point in 2D space."""

    x: float
    y: float

    def __getitem__(self, axis: int) -> float:
        """Allow indexing to access coordinates."""
        if axis == 0:
            return self.x
        elif axis == 1:
            return self.y
        raise IndexError("Point only has 2 dimensions (0 for x, 1 for y)")

    def as_tuple(self) -> Tuple[float, float]:
        """Return point as tuple for easier plotting."""
        return (self.x, self.y)


@dataclass
class Node:
    """A node in the KD-tree."""

    point: Point
    left: Optional[Node] = None
    right: Optional[Node] = None
    axis: int = 0
    label: str = ""  # Added label field

    def __post_init__(self):
        if not isinstance(self.axis, int) or self.axis not in (0, 1):
            raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis)")


class KDTree:
    """
    A 2D K-dimensional tree implementation for spatial partitioning.
    """

    def __init__(self, points: List[Point]):
        if not points:
            raise ValueError("Cannot create KD-tree with empty points list")
        self.points = points
        self.root: Optional[Node] = None
        self._next_label_idx = 0  # Track the next available label

    def _get_next_label(self) -> str:
        """Get the next available uppercase letter label."""
        if self._next_label_idx >= len(string.ascii_uppercase):
            return f"N{self._next_label_idx}"
        label = string.ascii_uppercase[self._next_label_idx]
        self._next_label_idx += 1
        return label

    def build(self) -> None:
        """Build the KD-tree from the points list."""
        self._next_label_idx = 0  # Reset label counter
        self.root = self._build_tree(self.points)

    def _build_tree(self, points: List[Point], depth: int = 0) -> Optional[Node]:
        if not points:
            return None

        axis = depth % 2
        points.sort(key=lambda p: p[axis])
        median_idx = len(points) // 2

        # Create node with median point and assign next label
        node = Node(point=points[median_idx], axis=axis, label=self._get_next_label())

        node.left = self._build_tree(points[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1 :], depth + 1)

        return node

    def visualize(
        self,
        figsize: Tuple[int, int] = (15, 6),
        point_size: int = 50,
        show_tree: bool = True,
        show_space: bool = True,
    ) -> None:
        if not self.root:
            raise ValueError("Tree not built yet. Call build() first.")

        if show_tree and show_space:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            self._draw_tree_structure(ax1)
            self._draw_space_partition(ax2, point_size)
        elif show_tree:
            fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
            self._draw_tree_structure(ax)
        elif show_space:
            fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
            self._draw_space_partition(ax, point_size)
        else:
            return

        plt.tight_layout()
        plt.show()

    def _draw_tree_structure(self, ax: Axes) -> None:
        def plot_node(
            node: Node,
            pos: Tuple[float, float] = (0, 0),
            level: int = 0,
            parent_pos: Optional[Tuple[float, float]] = None,
        ) -> None:
            if node is None:
                return

            x, y = pos[0], -level

            # Draw node
            ax.scatter(x, y, c="black", s=100)

            if parent_pos:
                ax.plot([parent_pos[0], x], [parent_pos[1], y], "gray", linestyle="-")

            # Add label and coordinates
            ax.annotate(
                f"{node.label}\n({node.point.x:.1f}, {node.point.y:.1f})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

            spacing = 2.0 / (2**level)
            plot_node(node.left, (x - spacing, y), level + 1, (x, y))
            plot_node(node.right, (x + spacing, y), level + 1, (x, y))

        ax.set_title("Tree Structure")
        plot_node(self.root)
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.3)

    def _draw_space_partition(self, ax: Axes, point_size: int) -> None:
        def draw_partition(
            node: Optional[Node], bounds: Tuple[float, float, float, float]
        ) -> None:
            if node is None:
                return

            xmin, xmax, ymin, ymax = bounds

            # Draw the splitting line with label
            if node.axis == 0:
                # Vertical line for x-axis split
                ax.vlines(
                    x=node.point.x,
                    ymin=ymin,
                    ymax=ymax,
                    colors="g",
                    linestyles="--",
                    alpha=0.5,
                )

                # Recurse on children with updated bounds
                draw_partition(node.left, (xmin, node.point.x, ymin, ymax))
                draw_partition(node.right, (node.point.x, xmax, ymin, ymax))
            else:
                # Horizontal line for y-axis split
                ax.hlines(
                    y=node.point.y,
                    xmin=xmin,
                    xmax=xmax,
                    colors="r",
                    linestyles="--",
                    alpha=0.5,
                )

                # Recurse on children with updated bounds
                draw_partition(node.left, (xmin, xmax, ymin, node.point.y))
                draw_partition(node.right, (xmin, xmax, node.point.y, ymax))

            # Add label near the point
            ax.text(
                node.point.x + 0.5,
                node.point.y + 0.25,
                f" {node.label}",
                verticalalignment="bottom",
                horizontalalignment="left",
            )

        # Get points array for plotting
        points_array = np.array([p.as_tuple() for p in self.points])

        # Calculate bounds with padding
        xmin, xmax = points_array[:, 0].min(), points_array[:, 0].max()
        ymin, ymax = points_array[:, 1].min(), points_array[:, 1].max()

        # Add padding
        padding = 0.1
        xpad = (xmax - xmin) * padding
        ypad = (ymax - ymin) * padding
        xmin, xmax = xmin - xpad, xmax + xpad
        ymin, ymax = ymin - ypad, ymax + ypad

        # Set the axis limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Draw the partition lines
        draw_partition(self.root, (xmin, xmax, ymin, ymax))

        # Plot the points
        ax.scatter(
            points_array[:, 0], points_array[:, 1], s=point_size, c="blue", alpha=0.6
        )

        ax.set_title("Space Partitioning")
        ax.grid(True, linestyle="--", alpha=0.3)


def generate_random_points(
    n: int,
    x_range: Tuple[float, float] = (0, 100),
    y_range: Tuple[float, float] = (0, 100),
    seed: Optional[int] = None,
) -> List[Point]:
    """
    Generate n random points within the specified ranges.
    """
    if seed is not None:
        np.random.seed(seed)

    return [
        Point(x=np.random.uniform(*x_range), y=np.random.uniform(*y_range))
        for _ in range(n)
    ]


def main():
    # Generate random points
    points = generate_random_points(20, seed=42)

    # Create and build the tree
    tree = KDTree(points)
    tree.build()

    # Visualize the tree
    tree.visualize(figsize=(15, 6), point_size=100)


if __name__ == "__main__":
    main()

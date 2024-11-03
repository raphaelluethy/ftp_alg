import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, left=None, right=None, axis=0):
        self.x = x
        self.y = y
        self.left = left
        self.right = right
        self.axis = axis


class KDTree:
    def __init__(self, points: list[Node]):
        self.points = points

    def build(self):
        self.root = self.build_tree(self.points, 0)

    def build_tree(self, points: list[Node], axis=0) -> Node:
        if len(points) == 0:
            return None
        if len(points) == 1:
            return points[0]
        axis = 0
        sorted_points = sort_nodes_by_axis(points, axis)
        median = sorted_points[len(sorted_points) // 2]
        left = list(
            filter(
                lambda node: get_axis_value(node, axis) < get_axis_value(median, axis),
                points,
            )
        )
        right = list(
            filter(
                lambda node: get_axis_value(node, axis) >= get_axis_value(median, axis),
                points,
            )
        )
        return Node(
            median.x,
            median.y,
            self.build_tree(left),
            self.build_tree(right),
            axis=1 if axis == 0 else 0,
        )

    def draw_points(self):
        plt.scatter(self.points[0].x, self.points[0].y)
        self.draw_points_of_tree(self.root)
        plt.show()

    def draw_points_of_tree(self, node: Node):
        if node.left is not None:
            self.draw_points_of_tree(node.left)
        plt.scatter(node.x, node.y)
        if node.right is not None:
            self.draw_points_of_tree(node.right)

        if node.axis == 0:
            plt.axvline(x=node.x, color="g", linestyle="--", alpha=0.3)
        else:
            plt.axhline(y=node.y, color="r", linestyle="--", alpha=0.3)

    def draw_tree(self):
        def plot_node(node, pos=(0, 0), level=0, parent_pos=None):
            if node is None:
                return

            x, y = pos[0], -level
            plt.scatter(x, y, c="black")
            if parent_pos:
                plt.plot([parent_pos[0], x], [parent_pos[1], y], "gray")

            plt.annotate(
                f"({node.x:.1f}, {node.y:.1f})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
            )

            spacing = 2.0 / (2**level)
            plot_node(node.left, (x - spacing, y), level + 1, (x, y))
            plot_node(node.right, (x + spacing, y), level + 1, (x, y))

        plt.figure(figsize=(12, 8))
        plot_node(self.root)
        plt.axis("equal")
        plt.show()


def get_axis_value(node: Node, axis: int) -> float:
    return node.x if axis == 0 else node.y


def sort_nodes_by_axis(nodes: list[Node], axis: int) -> list[Node]:
    """
    returns the median node
    axis 0 is x axis
    axis 1 is y axis
    """
    return sorted(nodes, key=lambda node: get_axis_value(node, axis))


def main():
    points = []
    for _ in range(20):  # creates 20 random points
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        points.append(Node(x, y))
    tree = KDTree(points)
    tree.build()
    tree.draw_tree()


if __name__ == "__main__":
    main()

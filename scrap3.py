import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. Function to manually select nodes with undo capability and zooming
def select_nodes(image, num_nodes):
    nodes = []

    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.title(f"Select {num_nodes} nodes by clicking (right-click to undo)")

    # Add zoom and pan functionality
    divider = make_axes_locatable(ax)
    zoom_tools = fig.canvas.toolbar  # Access the toolbar directly
    fig.canvas.toolbar_visible = True  # Ensure toolbar is visible

    def onclick(event):
        if len(nodes) < num_nodes and event.button == 1:  # Left-click to add node
            x, y = int(event.xdata), int(event.ydata)
            nodes.append((x, y))
            ax.plot(x, y, 'bo')
            ax.text(x, y, str(len(nodes) - 1), color='red', fontsize=12)
            fig.canvas.draw()
        elif event.button == 3 and nodes:  # Right-click to undo the last node
            nodes.pop()
            ax.clear()
            ax.imshow(image)
            for i, (x, y) in enumerate(nodes):
                ax.plot(x, y, 'bo')
                ax.text(x, y, str(i), color='red', fontsize=12)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return nodes

# 2. Function to select edges through console input
def select_edges(image, nodes):
    edges = []

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Plot nodes with labels
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'bo')
        ax.text(x, y, str(i), color='red', fontsize=12)

    def redraw_image():
        """Redraw the image with nodes and edges."""
        ax.clear()
        ax.imshow(image)
        # Draw edges
        for start, end in edges:
            pt1, pt2 = nodes[start], nodes[end]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-')
        # Draw nodes with labels
        for i, (x, y) in enumerate(nodes):
            ax.plot(x, y, 'bo')
            ax.text(x, y, str(i), color='red', fontsize=12)
        fig.canvas.draw()

    redraw_image()

    print(f"\nYou have selected {len(nodes)} nodes. Specify edges between nodes.")
    while True:
        edge_input = input("Enter 'start_node end_node' (e.g., '0 1'), 'u' to undo, or 'done' to finish: ").strip()

        if edge_input.lower() == 'done':
            break
        elif edge_input.lower() == 'u':  # Undo the last edge
            if edges:
                edges.pop()
                print("Last edge undone.")
            else:
                print("No edges to undo.")
            redraw_image()  # Redraw after undoing the edge
        else:
            try:
                start, end = map(int, edge_input.split())
                if 0 <= start < len(nodes) and 0 <= end < len(nodes) and start != end:
                    edges.append((start, end))
                    print(f"Edge {start} -> {end} added.")
                    redraw_image()  # Redraw after adding a new edge
                else:
                    print(f"Invalid edge. Ensure node indices are between 0 and {len(nodes) - 1} and not the same.")
            except ValueError:
                print("Invalid input. Enter two integers separated by a space.")

    print("Edge selection complete.")
    plt.show()

    return edges

def save_configuration(nodes, edges, filename='nodes_edges_8.json'):
    data = {'nodes': nodes, 'edges': edges}
    with open(filename, 'w') as f:
        json.dump(data, f)

# Example usage
image = plt.imread('./test_2.png')
num_nodes = int(input("Enter the number of nodes to select: "))
nodes = select_nodes(image, num_nodes)
edges = select_edges(image, nodes)

save_configuration(nodes, edges)

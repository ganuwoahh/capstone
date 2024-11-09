import os
import cv2
import json
import numpy as np

# 1. Function to manually select nodes with undo capability
def select_nodes(image, num_nodes):
    nodes = []
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(nodes) < num_nodes:
            nodes.append((x, y))
            draw_nodes(image, nodes)
        elif event == cv2.EVENT_RBUTTONDOWN and nodes:  # Undo the last node selection
            nodes.pop()
            draw_nodes(image, nodes)

    print(f"Select {num_nodes} nodes by left-clicking (Right-click to undo).")
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return nodes

def draw_nodes(image, nodes):
    """Redraw the nodes with their labels on the image."""
    temp_image = image.copy()
    for i, (x, y) in enumerate(nodes):
        cv2.circle(temp_image, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(temp_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", temp_image)

# 2. Function to dynamically select edges with undo capability
def select_edges(image, nodes):
    edges = []  # Store selected edges
    print(f"\nYou have selected {len(nodes)} nodes. Specify edges between nodes.")

    def redraw_image():
        """Redraw the image with nodes and edges."""
        temp_image = image.copy()
        # Draw edges
        for start, end in edges:
            pt1, pt2 = nodes[start], nodes[end]
            cv2.line(temp_image, pt1, pt2, (255, 0, 0), 2)
        # Draw nodes with labels
        for i, (x, y) in enumerate(nodes):
            cv2.circle(temp_image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(temp_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Image", temp_image)

    redraw_image()

    while True:
        try:
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
                cv2.waitKey(1)  # Refresh the image window
            else:
                start, end = map(int, edge_input.split())
                if 0 <= start < len(nodes) and 0 <= end < len(nodes) and start != end:
                    edges.append((start, end))
                    print(f"Edge {start} -> {end} added.")
                    redraw_image()  # Redraw after adding a new edge
                    cv2.waitKey(1)  # Refresh the image window
                else:
                    print(f"Invalid edge. Ensure node indices are between 0 and {len(nodes) - 1} and not the same.")
        except ValueError:
            print("Invalid input. Enter two integers separated by a space.")

    print("Edge selection complete.")
    cv2.imshow("Final Image with Edges", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges


# 3. Save the nodes and edges to a file
def save_configuration(nodes, edges, filename='nodes_edges_3.json'):
    data = {'nodes': nodes, 'edges': edges}
    with open(filename, 'w') as f:
        json.dump(data, f)

# 4. Load the configuration
def load_configuration(filename='nodes_edges_3.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

# 5. Function to extract pixel values along a line (for edge color)
def extract_edge_color(image, pt1, pt2):
    x_values, y_values = np.linspace(pt1[0], pt2[0], 100).astype(int), np.linspace(pt1[1], pt2[1], 100).astype(int)
    colors = image[y_values, x_values]
    avg_color = np.mean(colors, axis=0)
    return avg_color

# 6. Apply node and edge configuration to screenshots
def process_screenshots(screenshot_dir, config_filename='nodes_edges.json'):
    nodes, edges = load_configuration(config_filename)
    screenshots = [os.path.join(screenshot_dir, img) for img in os.listdir(screenshot_dir) if img.endswith('.png')]

    for screenshot in screenshots:
        image = cv2.imread(screenshot)
        result = image.copy()

        for edge in edges:
            pt1 = tuple(nodes[edge[0]])
            pt2 = tuple(nodes[edge[1]])
            color = extract_edge_color(image, pt1, pt2)
            print(f"Edge {pt1} -> {pt2}: Average Color = {color}")

# Main function to run the process
if __name__ == '__main__':
    # Step 1: Load an image and specify the number of nodes
    image_path = r'D:\code_stuff\cap\test_2.png'
    image = cv2.imread(image_path)
    num_nodes = int(input("Enter the number of nodes to select: "))

    # Step 2: Select nodes with undo capability
    node_image = image.copy()
    nodes = select_nodes(node_image, num_nodes)

    # Step 3: Select edges with undo capability
    edge_image = image.copy()
    edges = select_edges(edge_image, nodes)

    # Step 4: Save the configuration
    save_configuration(nodes, edges)

    # Step 5: Process all screenshots
    #screenshot_dir = os.path.dirname(image_path)
    #process_screenshots(screenshot_dir)

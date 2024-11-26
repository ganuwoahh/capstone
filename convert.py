import pandas as pd
import numpy as np

# Define the representative colors
colors = {
    "red": np.array([0, 0, 255]),         # BGR equivalent of RGB [255, 0, 0]
    "orange": np.array([0, 165, 255]),   # BGR equivalent of RGB [255, 165, 0]
    "green": np.array([0, 128, 0]),      # BGR equivalent of RGB [0, 128, 0]
    "yellow": np.array([0, 255, 255])    # BGR equivalent of RGB [255, 255, 0]
}


# Function to find the closest color
def classify_color(rgb_value):
    min_distance = float('inf')
    closest_color = None
    rgb_array = np.array(rgb_value)
    
    for color_name, color_rgb in colors.items():
        # Calculate Euclidean distance
        distance = np.linalg.norm(rgb_array - color_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color

# Load your data (replace 'your_file.xlsx' with the path to your file)
df = pd.read_excel('screenshot_data_chopped.xlsx')

# Identify edge columns (assuming they start with 'edge_' prefix)
edge_columns = [col for col in df.columns if col.startswith('edge_')]

# Apply color classification to each edge column
for col in edge_columns:
    # Convert string representation of list to actual list of floats
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df[col] = df[col].apply(classify_color)

# Save or display the modified DataFrame
print(df.head())
# Optionally, save the result to a new Excel file
df.to_excel('screenshot_data_color.xlsx', index=False)

import os
import time
import requests
import pandas as pd
import openpyxl
from scipy.stats import mode
from collections import Counter
from datetime import datetime
from selenium import webdriver
import numpy as np
import cv2
import json
import signal
import sys

# Function to get weather data
def get_weather_data():
    latitude = 12.906921
    longitude = 77.576487
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,windspeed_10m,pressure_msl,relative_humidity_2m",
        "current_weather": True,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params)
    data = response.json()

    current_weather = data['current_weather']
    temperature = current_weather['temperature']
    windspeed = current_weather['windspeed']
    winddirection = current_weather['winddirection']
    weathercode = current_weather['weathercode']

    hourly_data = data['hourly']
    hourly_times = hourly_data['time']

    hourly_times_dt = [datetime.fromisoformat(t) for t in hourly_times]
    closest_time_index = min(range(len(hourly_times_dt)), key=lambda i: abs(hourly_times_dt[i] - datetime.now()))

    precipitation = hourly_data['precipitation'][closest_time_index]
    pressure = hourly_data['pressure_msl'][closest_time_index]

    weather_data = {
        'Temperature': temperature,
        'Wind Speed': windspeed,
        'Wind Direction': winddirection,
        'Weather Code': weathercode,
        'Precipitation': precipitation,
        'Pressure': pressure
    }

    return weather_data

# Function to load configuration from a JSON file
def load_configuration(filename='nodes_edges_5.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

# Function to extract pixel values along a line (to check edge color) with color filtering
# Function to extract pixel values along a line (to check edge color) with color filtering
def extract_edge_color(image, pt1, pt2):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for filtering (HSV) and their respective solid colors in HSV
    color_ranges = {
        "lime_green": ((40, 100, 100), (70, 255, 255), [50, 255, 255]),  # Solid lime green
        "yellow": ((20, 100, 100), (30, 255, 255), [25, 255, 255]),      # Solid yellow
        "red": ((0, 100, 100), (10, 255, 255), [0, 255, 255]),           # Solid red
        "maroon": ((160, 100, 100), (179, 255, 255), [170, 255, 255])    # Solid maroon
    }

    # Initialize a combined mask
    combined_mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
    
    # Apply each color mask and combine
    for color, (lower, upper, _) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Extract the pixel values along the line
    x_values, y_values = np.linspace(pt1[0], pt2[0], 100).astype(int), np.linspace(pt1[1], pt2[1], 100).astype(int)
    masked_colors = hsv_image[y_values, x_values]  # Pixels along the line in HSV

    # Apply the mask to get only the specified colors; if no colors match, use the unmasked colors
    valid_masked_colors = masked_colors[combined_mask[y_values, x_values] > 0]

    # Calculate the average color
    if len(valid_masked_colors) > 0:
        avg_color = np.mean(valid_masked_colors, axis=0)
        # Check if avg_color falls within any of the predefined color ranges
        for color_name, (lower, upper, solid_color) in color_ranges.items():
            if all(lower[i] <= avg_color[i] <= upper[i] for i in range(3)):
                return solid_color  # Return the solid color for this range
        # If avg_color doesn't match any range, return the calculated average color
        return avg_color.tolist()
    else:
        # If no colors were detected in the specified ranges, return the unmasked average color
        avg_color = np.mean(masked_colors, axis=0)
        return avg_color.tolist()

# Function to highlight nodes and edges
def highlight_nodes_edges(image, nodes, edges):
    for i, (x, y) in enumerate(nodes):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green color for nodes
        cv2.putText(image, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for edge in edges:
        pt1 = tuple(nodes[edge[0]])
        pt2 = tuple(nodes[edge[1]])
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Blue color for edges

# Function to store images data in an Excel file
def store_images_data(image_path, weather_data, edge_data):
    filename = os.path.basename(image_path)
    date_time_str = filename.split('_')[1] + ' ' + filename.split('_')[2].split('.')[0]
    date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H-%M-%S')

    row_data = {
        'Date': date_time.date(),
        'Time': date_time.time(),
    }
    row_data.update(weather_data)
    row_data.update(edge_data)

    df_row = pd.DataFrame([row_data])

    file_exists = os.path.exists('testdata4.xlsx')

    if file_exists:
        with pd.ExcelWriter('testdata4.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            last_row = openpyxl.load_workbook('testdata4.xlsx')['Sheet1'].max_row
            df_row.to_excel(writer, index=False, header=False, startrow=last_row)
    else:
        with pd.ExcelWriter('testdata4.xlsx', engine='openpyxl') as writer:
            df_row.to_excel(writer, index=False)

    print(f"Data for {filename} saved to 'testdata4.xlsx'.")

# Function to take a screenshot and process it
def take_screenshot_and_process(url, interval):
    screenshots_dir = os.path.join(os.getcwd(), "screenshots/test10")
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    driver = webdriver.Chrome()

    try:
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)

        weather_data = get_weather_data()
        last_weather_update_time = datetime.now()

        nodes, edges = load_configuration()

        while True:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshot_{current_datetime}.png"
            save_path = os.path.join(screenshots_dir, filename)
            driver.save_screenshot(save_path)
            print(f"Screenshot saved to: {save_path}")

            image = cv2.imread(save_path)
            analysis_image = image.copy()

            edge_data = {}
            for edge in edges:
                pt1 = tuple(nodes[edge[0]])
                pt2 = tuple(nodes[edge[1]])
                avg_color = extract_edge_color(analysis_image, pt1, pt2)
                edge_identifier = f"edge_{edge[0]}_{edge[1]}"
                edge_data[edge_identifier] = avg_color

            highlight_nodes_edges(image, nodes, edges)
            modified_save_path = os.path.join(screenshots_dir, f"highlighted_{filename}")
            cv2.imwrite(modified_save_path, image)
            print(f"Highlighted image saved to: {modified_save_path}")

            store_images_data(save_path, weather_data, edge_data)

            if (datetime.now() - last_weather_update_time).seconds >= 3600:
                weather_data = get_weather_data()
                last_weather_update_time = datetime.now()

            time.sleep(interval)

    finally:
        driver.quit()

# Signal handler to gracefully exit on Ctrl+C
def signal_handler(sig, frame):
    print("\nInterrupted! Exiting...")
    sys.exit(0)

if __name__ == "__main__":
    url = "https://www.google.com/maps/@12.9065263,77.5729105,17.1z/data=!5m1!1e1?entry=ttu"
    interval = 30
    signal.signal(signal.SIGINT, signal_handler)
    take_screenshot_and_process(url, interval)

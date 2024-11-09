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
    # Define the endpoint and parameters for weather data
    latitude = 12.906921
    longitude = 77.576487
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,windspeed_10m,pressure_msl,relative_humidity_2m",
        "current_weather": True,
        "timezone": "Asia/Kolkata"  # Specify your timezone if needed
    }

    # Make the request
    response = requests.get(url, params=params)
    data = response.json()

    # Extract current weather data
    current_weather = data['current_weather']
    temperature = current_weather['temperature']
    windspeed = current_weather['windspeed']
    winddirection = current_weather['winddirection']
    weathercode = current_weather['weathercode']

    # Extract hourly data
    hourly_data = data['hourly']
    hourly_times = hourly_data['time']

    # Convert hourly times to datetime objects and find the closest time
    hourly_times_dt = [datetime.fromisoformat(t) for t in hourly_times]
    closest_time_index = min(range(len(hourly_times_dt)), key=lambda i: abs(hourly_times_dt[i] - datetime.now()))

    # Extract precipitation and pressure for the closest time
    precipitation = hourly_data['precipitation'][closest_time_index]
    pressure = hourly_data['pressure_msl'][closest_time_index]

    # Combine all weather data into a dictionary
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
def load_configuration(filename='nodes_edges_2.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

# Function to extract pixel values along a line (to check edge color)
import numpy as np
from scipy.stats import mode

import numpy as np
from scipy.stats import mode

def extract_edge_color(image, pt1, pt2):
    x_values, y_values = np.linspace(pt1[0], pt2[0], 100).astype(int), np.linspace(pt1[1], pt2[1], 100).astype(int)
    colors = image[y_values, x_values]
    avg_color = np.mean(colors, axis=0)
    return avg_color



def highlight_nodes_edges(image, nodes, edges):
    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green color for nodes
        cv2.putText(image, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw edges
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

    file_exists = os.path.exists('testdata.xlsx')

    if file_exists:
        with pd.ExcelWriter('testdata.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            last_row = openpyxl.load_workbook('testdata.xlsx')['Sheet1'].max_row
            df_row.to_excel(writer, index=False, header=False, startrow=last_row)
    else:
        with pd.ExcelWriter('testdata.xlsx', engine='openpyxl') as writer:
            df_row.to_excel(writer, index=False)

    print(f"Data for {filename} saved to 'testdata.xlsx'.")

import cv2

def take_screenshot_and_process(url, interval):
    screenshots_dir = os.path.join(os.getcwd(), "screenshots/test5")
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    driver = webdriver.Chrome()

    try:
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)

        # Initialize weather data
        weather_data = get_weather_data()
        last_weather_update_time = datetime.now()

        # Load nodes and edges configuration
        nodes, edges = load_configuration()

        while True:  # Infinite loop to take screenshots until interrupted
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshot_{current_datetime}.png"
            save_path = os.path.join(screenshots_dir, filename)
            driver.save_screenshot(save_path)
            print(f"Screenshot saved to: {save_path}")

            # Process the screenshot
            image = cv2.imread(save_path)
            
            # Resize the image by the factor of 12048/17376
            scaling_factor = 12048 / 17376
            new_width = int(image.shape[1] * scaling_factor)
            new_height = int(image.shape[0] * scaling_factor)
            resized_image = cv2.resize(image, (new_width, new_height))

            # Highlight nodes and edges
            highlight_nodes_edges(resized_image, nodes, edges)
            modified_save_path = os.path.join(screenshots_dir, f"highlighted_{filename}")
            cv2.imwrite(modified_save_path, resized_image)
            print(f"Highlighted image saved to: {modified_save_path}")
            
            edge_data = {}
            for edge in edges:
                pt1 = tuple(nodes[edge[0]])
                pt2 = tuple(nodes[edge[1]])
                avg_color = extract_edge_color(resized_image, pt1, pt2)
                edge_identifier = f"edge_{edge[0]}_{edge[1]}"
                edge_data[edge_identifier] = avg_color.tolist()

            # Store the image data with the current weather data
            store_images_data(save_path, weather_data, edge_data)

            # Check if an hour has passed since the last weather update
            if (datetime.now() - last_weather_update_time).seconds >= 3600:
                weather_data = get_weather_data()  # Fetch new weather data
                last_weather_update_time = datetime.now()

            time.sleep(interval)

    finally:
        driver.quit()


# Signal handler to gracefully exit on Ctrl+C
def signal_handler(sig, frame):
    print("\nInterrupted! Exiting...")
    sys.exit(0)

# Main function to run the process
if __name__ == "__main__":
    url = "https://www.google.com/maps/@12.9065263,77.5729105,17.1z/data=!5m1!1e1?entry=ttu"  # Replace with your target URL
    interval = 30  # Interval in seconds between screenshots
    signal.signal(signal.SIGINT, signal_handler)  # Register signal handler for Ctrl+C
    take_screenshot_and_process(url, interval)
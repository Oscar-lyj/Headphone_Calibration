import os
import csv
import cv2
import numpy as np
from PIL import Image

# Global variables
img = None  
sens = 60  # Default sensitivity

# Save path
SAVE_PATH = "D:\\李彦君\\其他\\coding\\headphone_calibration\\app\\static\\uploads"

def extract_curve(image, selected_region, sens):
    """Extract the curve from the selected bounding box (Mode 1)."""
    x1, y1, x2, y2 = selected_region
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, sens, 255, cv2.THRESH_BINARY_INV)
    curve_points = np.column_stack(np.where(thresholded > 0))
    return curve_points

def extract_curve_by_color(image, selected_region, sens):
    """Extract curve points by matching the selected color (Mode 2)."""
    selected_color = selected_region  # The color passed in `selected_region`
    color_diff = np.linalg.norm(image - selected_color, axis=2)
    threshold = sens
    color_points = np.column_stack(np.where(color_diff < threshold))
    return color_points

def map_frequency_to_x(img_width, start_freq=20, end_freq=20000, freq_step=1.01):
    """Map x-coordinates to logarithmic frequency scale."""
    frequencies, x_positions = [], []
    current_freq = start_freq
    while current_freq <= end_freq:
        frequencies.append(current_freq)
        x_pos = int(np.log10(current_freq / start_freq) * img_width / np.log10(end_freq / start_freq))
        x_positions.append(x_pos)
        current_freq *= freq_step
    return frequencies, x_positions

def interpolate_curve_points(curve_points, x_positions):
    """Interpolate missing amplitude values for frequency mapping."""
    curve_points = np.array(curve_points)
    amplitudes = []

    for x in x_positions:
        y_values = curve_points[curve_points[:, 1] == x][:, 0]
        amplitudes.append(np.mean(y_values) if len(y_values) > 0 else np.nan)

    amplitudes = np.array(amplitudes)
    nan_indices = np.isnan(amplitudes)
    if np.any(nan_indices):
        amplitudes[nan_indices] = np.interp(
            np.flatnonzero(nan_indices),
            np.flatnonzero(~nan_indices),
            amplitudes[~nan_indices]
        )
    return amplitudes

def save_to_csv(frequencies, amplitudes, saving_type, reference_mode):
    """Save extracted curve data to a CSV file."""
    if saving_type == 1:
        filename="extracted_curve.csv"
    elif saving_type == 2:
        filename ="extracted_reference_curve.csv"
        if reference_mode == 5: #fliping amplitute for csv file
            amplitudes = -amplitudes
    filepath = os.path.join(SAVE_PATH, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["raw"])
        for amp in amplitudes:
            writer.writerow([amp])
    print(f"Saved CSV to: {filepath}")

def interactive_image_selection(image_path, mode, sens, max_pos, min_pos, selected_region, file_type):
    # if filetype = 1, it is used in the upload png file
    # if filetype = 2, it is used in extracting reference file
    """Process curve extraction based on the selected region or color."""
    global img

    # Load image
    try:
        pil_image = Image.open(image_path)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Ask for amplitude scaling range
    y_min = float(min_pos)
    y_max = float(max_pos)

    # Process the selected region based on mode
    if mode == 1:
        # Mode 1: Extract using the bounding box
        curve_points = extract_curve(img, selected_region, sens)
    elif mode == 2:
        # Mode 2: Extract using the selected color
        curve_points = extract_curve_by_color(img, selected_region, sens)
        
    else:
        print("Invalid mode selected.")
        return

    if curve_points is None or len(curve_points) == 0:
        print("No curve extracted.")
        return

    # Map curve to frequency
    frequencies, x_positions = map_frequency_to_x(img.shape[1])
    amplitudes = interpolate_curve_points(curve_points, x_positions)

    # Scale and save results
    scaled_amplitudes = ((amplitudes - np.min(amplitudes)) / (np.max(amplitudes) - np.min(amplitudes))) * (y_max - y_min) + y_min
    save_to_csv(frequencies, -scaled_amplitudes, file_type, mode)
    # return frequencies, scaled_amplitudes

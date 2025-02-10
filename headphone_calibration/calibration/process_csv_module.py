# using the model to return frequency response
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os


# defining the constants
f = pd.read_csv("headphone_calibration\\calibration\\reference.csv")
freqs = f["frequency"].values # defining the frequency

# defining the reference curves
over_ear_ref = f["over ear"].values
over_ear_no_bass_ref = f["over ear w/o bass"].values
in_ear_ref = f["in ear"].values
in_ear_no_bass_ref = f["in ear w/o bass"].values



# function to predict the frequency response array from an image
'''
def freq_response_AI(image_path, model):
    img = load_img(image_path, color_mode="grayscale", target_size = (128,128))
    img_array = img_to_array(img)/255.0 # normalization
    img_array = np.expand_dims(img_array,axis = 0) # add batch dimension
    
    freq = model.predict(img_array)
    return freq.flattern()
'''


# using n bands of eq to calibrate the object frequency response to reference frequency response

import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, amp, mean, stddev):
    # gaussian function
    return amp*np.exp(-((x-mean)**2)/(2*stddev**2))

def cal_reference(curve_type, path = None, mode = None):
    # return the reference curve for the need
    # 0: in ear with bass
    # 1: in ear without bass
    # 2: over ear with bass
    # 3: over ear without bass
    # 4: upload frequency curve, using the path for the image in type jpg or png
    # 5: upload frequency data, using path for the file type of csv 
    
    if curve_type == 0:
        return in_ear_ref
    elif curve_type == 1:
        return in_ear_no_bass_ref
    elif curve_type == 2:
        return over_ear_ref
    elif curve_type == 3:
        return over_ear_no_bass_ref
    elif curve_type == 4:
        df = pd.read_csv(path)
        reference_curve = df["raw"].values  
        return reference_curve
    elif curve_type ==5:
        df = pd.read_csv(path)
        reference_curve = df["raw"].values  
        return reference_curve

def fit_gaussians(freqs, difference, num_bands):
    # fit multiple gaussian curves to the difference curve formed by the reference curve and the object curve
    
    # parameter:
    # difference: an array of difference between the reference and object curve
    # num_bands: Number of EQ bands
    
    # return:
    # a list of parameters of gaussian curves [amp, mean, stddev] for each band
    
    # initial guess for gaussian parameters
    initial_params = []
    step = len(freqs)//num_bands
    for i in range(num_bands):        
        # deviding the frequency range into n segments where n = number of bands
        start_idx = i*step
        end_idx = min(len(freqs),(i+1)*step)
        segment = difference[start_idx:end_idx]
        freq_segment = freqs[start_idx:end_idx]
        
        # estimating the initial parameters for the segment
        amp = np.max(segment) if np.max(segment) > np.abs(np.min(segment)) else np.min(segment)
        mean = freq_segment[np.argmax(np.abs(segment))]
        stddev = (freq_segment[-1]-freq_segment[0])/2
        initial_params.extend([amp,mean,stddev])
        
    # fit multiple gaussians
    def multi_gaussian(x,*params):
        y = np.zeros_like(x)
        for i in range(num_bands):
            amp, mean, stddev = params[i * 3 : i * 3 + 3]
            y+= gaussian(x, amp, mean, stddev)
        return y
    
    bounds = (
        [-np.inf] * len(initial_params), # lower bounds
        [np.inf] * len(initial_params), # upper bounds
    )
    params, _ = curve_fit(multi_gaussian, freqs, difference, p0=initial_params, bounds=bounds, maxfev=1000000)
    return params.reshape(num_bands, 3)

matplotlib.use('Agg')  # Use a non-interactive backend

def apply_gaussian_eq(object_array, freqs, gaussian_params):
    # applying the gaussian correction to the object array
    correction = np.zeros_like(object_array)
    for amp, mean, stddev in gaussian_params:
        correction += gaussian(freqs, amp, mean, stddev)

    corrected_array = object_array + correction
    '''
    # Plot the results and save to a file
    plt.gca().set_box_aspect(0.35)
    plt.xscale("log")
    plt.plot(freqs, correction, label="Correction Curve")
    plt.plot(freqs, object_array, color="red", label="Original Curve")
    plt.plot(freqs, corrected_array, color="darkblue", label="Corrected Curve")
    plt.xlabel("Frequency (Hz, log scale)")
    plt.ylabel("Intensity (dB)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title("Frequency Response Correction (Gaussian Fit)")

    # Save the plot instead of showing it
    plt.close()  # Close the figure
    '''
    return corrected_array

def cal_eq_gaussians(object_array, reference_array, freqs, num_bands):
    # eq calibration by fitting and applying the gaussian correction
          
    difference = reference_array - object_array
    
    # fit the gaussians to the difference curve
    gaussian_params = fit_gaussians(freqs, difference, num_bands)
    
    # apply the gaussian corrections
    corrected_array = apply_gaussian_eq(object_array, freqs, gaussian_params)
    
    # extracting eq parameters
    eq_bands = []
    for amp, mean, stddev in gaussian_params:
        q_value = mean / (2 * stddev)
        eq_band = {
            "frequency": int(mean),
            "gain": round(amp,1),
            "filter_type": "bell",
            "Q_value": round(q_value,2),
        }
        eq_bands.append(eq_band)    
        
    # Visualization
    plt.gca().set_box_aspect(0.35)
    plt.xscale("log")
    plt.plot(freqs, reference_array, linestyle="--", color="dimgrey", label="Reference Frequency Response")
    plt.plot(freqs, object_array, color="red", label="Original Frequency Response")
    plt.plot(freqs, corrected_array, color="darkblue", label="Corrected Frequency Response")
    plt.ylim(-60,25)
    plt.xlabel("Frequency (Hz, log scale)")
    plt.ylabel("Intensity (dB)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title("Frequency Response Correction (Gaussian Fit)")
    plt.show()
    graph_path = os.path.join('headphone_calibration\\app\\static\\uploads', 'corrected_response.png')
     # Save plot as a file in the static folder
    # plt.savefig(graph_path)
    plt.close()

    return eq_bands, graph_path, corrected_array

def match_levels(object_array, reference_array):
    """ Adjust object curve to match the reference curve's overall level """
    object_mean = np.mean(object_array)
    reference_mean = np.mean(reference_array)
    level_adjustment = reference_mean - object_mean
    return (object_array + level_adjustment)

# load the model
# model = load_model("frequency_response_model.h5")

# to convert the image to array of frequency
# image_path = ""
# frequency_intensity = freq_response(image_path,model)
def process_csv(file_path, bands, reference_type, reference_path = None, reference_mode = None):
    # Read the CSV and process it
    df = pd.read_csv(file_path)
    object_curve = df["raw"].values
    reference_curve = cal_reference(reference_type, reference_path, reference_mode)  
    object_curve = match_levels(object_curve, reference_curve)
    eq_bands, graph_path, corrected_curve = cal_eq_gaussians(object_curve, reference_curve, freqs, bands)

    return eq_bands, graph_path, object_curve, reference_curve, corrected_curve, freqs





import cv2
import numpy as np
import os
from scipy.signal import resample, welch
from scipy.signal.windows import hann
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from debug import plot_colored_edges
import argparse

def reorderlines(bou_list, p0):
    # Placeholder: return boundary with closest starting point to p0
    if not bou_list:
        return [], p0
    closest_idx = np.argmin([np.linalg.norm(np.array(b[0]) - p0) for b in bou_list])
    bou = bou_list[closest_idx]
    return [bou], np.array(bou[0])

def preprocess(aspect_ratio, points_per_scan, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (1024, int(1024 / aspect_ratio))) #resize with aspect ratio
    blurred = cv2.GaussianBlur(resized, (3, 3), 1024 / points_per_scan) # Gaussian blur vary with scan quality
    # the more points per scan, the less sigma therefore sharper edges but more noise
    _, otsued = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding to get binary image
    edges = cv2.Canny(np.uint8(otsued), 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if (contours is not None) & (len(contours) != 0):
        print("contours found")
    #     plot_colored_edges(resized, contours)  # Debugging: visualize contours
    else:
        print("No contours found")
        print("using empty frame")
        # create an empty frame
        empty_frame = np.zeros_like(edges)
        # draw a white rectangle around the edge of the empty frame
        cv2.rectangle(empty_frame, (0, 0), (empty_frame.shape[1], empty_frame.shape[0]), (255, 255, 255), 1)
        # find contours in the empty frame
        contours, _ = cv2.findContours(empty_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Save the processed frame for debugging

    output_dir = os.path.join(os.getcwd(), "output_frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_count = len([name for name in os.listdir(output_dir) if name.startswith("frame_") and name.endswith(".png")])

    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count + 1}.png"), otsued)
    return contours

def parse_arguments():
    """
    Parse command-line arguments for video processing.

    Returns:
        tuple: A tuple containing the video path, output directory, and output WAV file path.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a video file and generate audio output.")
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    
    # Parse arguments
    args = parser.parse_args()
    vid_path = args.video
    output_dir = args.output_dir

    # debug
    # vid_path = "Test_Vid\Jamiroquai - Virtual Insanity (Official Video) [4JkIs37a2JE].mp4"
    # output_dir = "Output"

    # Validate video file
    if not os.path.isfile(vid_path):
        raise ValueError(f"Video file '{vid_path}' does not exist.")

    # Validate output directory
    if not os.path.isdir(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Do you want to create it? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            os.makedirs(output_dir)
            return vid_path, output_dir
        else:
            raise ValueError(f"Output directory '{output_dir}' does not exist and was not created.")
    return vid_path, output_dir


vid_path, output_dir = parse_arguments()

# Generate WAV file path
vid_name = os.path.splitext(os.path.basename(vid_path))[0]
wav_path = os.path.join(output_dir, vid_name + ".wav")

print("Loading file...")

# Load video
Vid = cv2.VideoCapture(vid_path)
vidFrameRate = Vid.get(cv2.CAP_PROP_FPS)
nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
vidWidth = int(Vid.get(cv2.CAP_PROP_FRAME_WIDTH))
vidHeight = int(Vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Parameters
scanPerFrame = 2
sample_freq = 48000
points_per_frame = round(sample_freq / vidFrameRate) # Points per frame
point_per_scan = round(points_per_frame / scanPerFrame) # Points per scan
aspect_ratio = vidWidth / vidHeight

# Processing frames
print("Processing frames...")
bouDotxy = []

init_point = np.array([(1024 / aspect_ratio + 1) / 2, (1024 + 1) / 2]) # initial point 

for k in range(nFrames):
    ret, frame = Vid.read()
    print(f"Frame {k + 1}/{nFrames}")
 
    if not ret:
        break

    contours = preprocess(aspect_ratio, point_per_scan, frame)

    Bou = [cnt[:, 0, :] for cnt in contours if len(cnt) > 0] # filter out empty contours

    BouTemp, init_point = reorderlines(Bou, init_point)
    
    if len(BouTemp) == 0:
        init_point = np.array([(1024 / aspect_ratio + 1) / 2, (1024 + 1) / 2])
        bouDotTemp = np.full((points_per_frame, 2), np.nan)
    else:
        bouDot = np.vstack(BouTemp)
        bouDot = resample(bouDot, point_per_scan, axis=0)
        bouDotTemp = np.tile(bouDot, (scanPerFrame, 1))

    bouDotxy.append(bouDotTemp)

Vid.release()

# Adjust amplitude
print("Adjusting amplitude...")
bouDotxy = np.vstack(bouDotxy) # Stack all frames
bouDotxy = bouDotxy - np.nanmean(bouDotxy, axis=0) # subtract mean along x
bouDotxy = bouDotxy / np.nanmax(np.abs(bouDotxy)) # divide by the absolute max value
# bouDotxy[:, 0] = -bouDotxy[:, 0] # invert x axis
# bouDotxy = bouDotxy[:, [1, 0]] # swap x and y axis
bouDotxy[np.isnan(bouDotxy)] = 0 # replace NaN with 0

# PSD plot
# print("Drawing PSD...")
# winlen = 2 * Fs
# window = hann(winlen, sym=False)
# noverlap = winlen // 2
# nfft = winlen
# f, pxx = welch(bouDotxy, fs=Fs, window=window, noverlap=noverlap, nfft=nfft, axis=0)
# plt.semilogx(f, pxx)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power")
# plt.title("Power Spectral Density")
# plt.grid()
# plt.show()

# Output audio
print("Output...")
sf.write(wav_path, bouDotxy, int(sample_freq))

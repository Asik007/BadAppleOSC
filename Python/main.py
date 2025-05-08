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

def reorderlines(bou_list, p0):
    # Placeholder: return boundary with closest starting point to p0
    if not bou_list:
        return [], p0
    closest_idx = np.argmin([np.linalg.norm(np.array(b[0]) - p0) for b in bou_list])
    bou = bou_list[closest_idx]
    return [bou], np.array(bou[0])

# Parameters
scanPerFrame = 2
sample_freq = 48000

# File dialogs
root = tk.Tk()
root.withdraw()
# vid_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mpg *.wmv *.mp4 *.m4v *.mov *.mkv")])
vid_path = r"C:\Users\Dynames\Code\BadAppleOSC\Test_Vid\Touhou - Bad Apple.mp4"
vid_name = os.path.splitext(os.path.basename(vid_path))[0]
wav_path = filedialog.askdirectory(title="Select output directory") + "/" + vid_name + ".wav"
# Read video
print("Loading file...")
Vid = cv2.VideoCapture(vid_path)
vidFrameRate = Vid.get(cv2.CAP_PROP_FPS)
nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
vidWidth = int(Vid.get(cv2.CAP_PROP_FRAME_WIDTH))
vidHeight = int(Vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
WHR = vidWidth / vidHeight

dotNumPF = round(sample_freq / vidFrameRate)
dotNum = round(dotNumPF / scanPerFrame)

# Processing frames
print("Processing frames...")
bouDotxy = []
p0 = np.array([(1024 / WHR + 1) / 2, (1024 + 1) / 2])

for k in range(nFrames):
    ret, frame = Vid.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (1024, int(1024 / WHR)))
    blurred = cv2.GaussianBlur(resized, (3, 3), 1024 / dotNum)
    _, otsued = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(np.uint8(otsued), 100, 200)

    # Find boundaries
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if (contours is not None) & (len(contours) != 0):
        print("No contours found")
        plot_colored_edges(resized, contours)  # Debugging: visualize contours

    Bou = [cnt[:, 0, :] for cnt in contours if len(cnt) > 0]



    BouTemp, p0 = reorderlines(Bou, p0)
    
    if len(BouTemp) == 0:
        p0 = np.array([(1024 / WHR + 1) / 2, (1024 + 1) / 2])
        bouDotTemp = np.full((dotNumPF, 2), np.nan)
    else:
        bouDot = np.vstack(BouTemp)
        bouDot = resample(bouDot, dotNum, axis=0)
        bouDotTemp = np.tile(bouDot, (scanPerFrame, 1))

    bouDotxy.append(bouDotTemp)

Vid.release()

# Adjust amplitude
print("Adjusting amplitude...")
bouDotxy = np.vstack(bouDotxy)
bouDotxy = bouDotxy - np.nanmean(bouDotxy, axis=0)
bouDotxy = bouDotxy / np.nanmax(np.abs(bouDotxy))
bouDotxy[:, 0] = -bouDotxy[:, 0]
bouDotxy = bouDotxy[:, [1, 0]]
bouDotxy[np.isnan(bouDotxy)] = 0

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

from tkinter import filedialog
import tkinter as tk
import wave
import numpy as np
import time
import matplotlib.pyplot as plt

def plot_LR_xy(left_channel, right_channel):
    
    if np.sum(left_channel) == np.nan or np.sum(right_channel) == np.nan:
        print("channel is NaN")
        return
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Left vs Right Audio Channels")
    ax.set_xlabel("Left Channel")
    ax.set_xlim(np.min(left_channel), np.max(left_channel))
    ax.set_ylim(np.min(right_channel), np.max(right_channel))
    ax.set_ylabel("Right Channel")
    ax.grid(True)

    # Plot points incrementally
    scatter = ax.plot(left_channel, right_channel, 'bo--', alpha=0.7)[0]
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.waitforbuttonpress()
    plt.close(fig)  # Close the figure after plotting
    return

def plot_audio_as_xy(audio_file):
    # Open the audio file
    with wave.open(audio_file, 'rb') as wav:
        # Ensure the file is stereo
        if wav.getnchannels() != 2:
            raise ValueError("Audio file must be stereo (2 channels).")
        samp_rate = wav.getframerate()  # Get the sample rate
        
        # Read audio frames and convert to numpy array
        for frame_start in range(0, wav.getnframes(), samp_rate):
            # Read a chunk of audio data
            wav.setpos(frame_start)
            if frame_start + samp_rate > wav.getnframes():
                break
            frame = wav.readframes(samp_rate)
            audio_data = np.frombuffer(frame)
            
            # Split into left and right channels
            left_channel = audio_data[0::2]
            right_channel = audio_data[1::2]
            
            plot_LR_xy(left_channel, right_channel)  # Call the plotting function
            # Normalize the data for better visualization
            
            # left_channel = left_channel / np.max(np.abs(left_channel))
            # right_channel = right_channel / np.max(np.abs(right_channel))
            
            # Plot the left channel as X and right channel as Y
            
            
# Example usage
root = tk.Tk()
root.withdraw()
audio_file_path = filedialog.askopenfilename(filetypes=[("Audio file", "*.wav")])
plot_audio_as_xy(audio_file_path)
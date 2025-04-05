import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (adjust the path as needed for your mobile device)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "crossfade_output.avi")

# Collect image files with specified extensions from the Downloads folder
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()  # sort images alphabetically; adjust if needed

# Video parameters
fps = 30                               # Frames per second for the output video
hold_frames = int(2.5 * fps)           # Each image is held for 2.5 seconds
transition_frames = 30                 # Number of frames for the crossfade transition

# Ensure there is at least one image
if not image_files:
    raise ValueError("No images found in the specified folder.")

# Load the first image to determine video dimensions
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
height, width = first_img.shape[:2]

# Function to generate crossfade transition frames between two images
def crossfade_transition(img1, img2, num_frames):
    frames = []
    # Resize second image if needed to ensure both images have the same dimensions
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (width, height))
    # Generate frames with gradually changing blending weights
    for i in range(num_frames):
        # Calculate blending factor (alpha goes from 0 to 1)
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

# List to store all frames of the final video
all_frames = []

# Process each image: hold each image, then transition to the next if available
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image can't be read
    # Resize image if its size doesn't match the reference dimensions
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    
    # Hold the current image for the specified duration
    for _ in range(hold_frames):
        all_frames.append(img)
    
    # If not the last image, generate a crossfade transition to the next image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (height, width):
            next_img = cv2.resize(next_img, (width, height))
        transition = crossfade_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Create a VideoWriter object to compile frames into a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write all frames to the video file
for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Crossfade transition video saved to:", output_video_path)

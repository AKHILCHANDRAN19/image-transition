import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (adjust as needed for your mobile device)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "zoom_transition_output.avi")

# Collect image files with specified extensions from the Downloads folder
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()  # Sorting images alphabetically

if not image_files:
    raise ValueError("No images found in the specified folder.")

# Video parameters
fps = 30                              # Frames per second for the output video
hold_duration = 2.5                   # Each image is shown for 2.5 seconds
hold_frames = int(hold_duration * fps)  # Number of frames to hold each image
transition_frames = 30                # Total number of frames for the zoom transition
half_frames = transition_frames // 2  # Divide transition into two halves

# Load the first image to determine video dimensions
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
height, width = first_img.shape[:2]

# Zoom factor: at the peak of the zoom, the visible area will be 1/zoom_factor of the full size.
zoom_factor = 2.0

# Function to generate zoom in frames for an image.
# This gradually crops the image from full size to a smaller, center region.
def zoom_in_frames(img, num_frames, zoom_factor, width, height):
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)  # t goes from 0 (no zoom) to 1 (maximum zoom)
        # Calculate the new dimensions:
        new_w = int(width - t * (width - width / zoom_factor))
        new_h = int(height - t * (height - height / zoom_factor))
        # Determine coordinates for a centered crop:
        x1 = (width - new_w) // 2
        y1 = (height - new_h) // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        # Resize back to full dimensions:
        frame = cv2.resize(cropped, (width, height))
        frames.append(frame)
    return frames

# Function to generate zoom out frames for an image.
# This starts from a zoomed-in view (central region) and gradually reveals the full image.
def zoom_out_frames(img, num_frames, zoom_factor, width, height):
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)  # t goes from 0 (zoomed in) to 1 (full image)
        # Calculate dimensions: at t=0, size is width/zoom_factor; at t=1, it's full size.
        new_w = int(width / zoom_factor + t * (width - width / zoom_factor))
        new_h = int(height / zoom_factor + t * (height - height / zoom_factor))
        # Center the crop:
        x1 = (width - new_w) // 2
        y1 = (height - new_h) // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        frame = cv2.resize(cropped, (width, height))
        frames.append(frame)
    return frames

# List to hold all frames of the final video
all_frames = []

# Process each image:
# 1. Hold each image for a specified duration.
# 2. For each pair (if not the last image), generate a zoom transition where the current image zooms in and the next image zooms out.
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image cannot be read
    # Resize image if needed to match the reference dimensions
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    
    # Hold the current image for 2.5 seconds
    for _ in range(hold_frames):
        all_frames.append(img)
    
    # If this is not the last image, create the zoom transition:
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (height, width):
            next_img = cv2.resize(next_img, (width, height))
        # Generate the first half of the transition: zoom in on the current image
        zoom_in = zoom_in_frames(img, half_frames, zoom_factor, width, height)
        # Generate the second half: zoom out on the next image
        zoom_out = zoom_out_frames(next_img, half_frames, zoom_factor, width, height)
        # Combine both halves to form the complete transition
        transition = zoom_in + zoom_out
        all_frames.extend(transition)

# Write all frames to a video file using OpenCV's VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Zoom transition video saved to:", output_video_path)

import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (adjust as needed)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "blur_transition_output.avi")

# Collect image files with specified extensions from the Downloads folder
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()  # Sort images alphabetically; adjust if needed

if not image_files:
    raise ValueError("No images found in the specified folder.")

# Video parameters
fps = 30                              # Frames per second for the output video
hold_duration = 2.5                   # Each image is shown for 2.5 seconds
hold_frames = int(hold_duration * fps)  # Number of frames to hold each image
transition_frames = 30                # Number of frames for the blur transition

# Load the first image to determine video dimensions
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
height, width = first_img.shape[:2]

# Function to create blur transition frames between two images
def blur_transition(img1, img2, num_frames):
    frames = []
    max_kernel_size = 51  # Maximum kernel size for blurring; must be odd

    # Ensure both images have the same dimensions by resizing if needed
    if img1.shape[:2] != (height, width):
        img1 = cv2.resize(img1, (width, height))
    if img2.shape[:2] != (height, width):
        img2 = cv2.resize(img2, (width, height))

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        kernel_size = int(1 + alpha * (max_kernel_size - 1))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd

        # Apply increasing blur to img1
        blurred_img1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)

        # Apply decreasing blur to img2
        reverse_alpha = 1 - alpha
        reverse_kernel_size = int(1 + reverse_alpha * (max_kernel_size - 1))
        if reverse_kernel_size % 2 == 0:
            reverse_kernel_size += 1  # Ensure kernel size is odd
        blurred_img2 = cv2.GaussianBlur(img2, (reverse_kernel_size, reverse_kernel_size), 0)

        # Blend the two images
        blended_frame = cv2.addWeighted(blurred_img1, 1 - alpha, blurred_img2, alpha, 0)
        frames.append(blended_frame)

    return frames

# List to store all frames for the final video
all_frames = []

# Process each image: hold each image, then add blur transition to next image if available
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image cannot be read

    # Resize the image if necessary to match the reference dimensions
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))

    # Hold the current image for the specified duration (2.5 seconds)
    for _ in range(hold_frames):
        all_frames.append(img)

    # If not the last image, generate blur transition frames to the next image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (height, width):
            next_img = cv2.resize(next_img, (width, height))
        transition = blur_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Create a VideoWriter object to compile frames into a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write all frames to the video file
for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Blur transition video saved to:", output_video_path)

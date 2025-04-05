import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (modify as needed)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "transition_output.avi")

# Collect image files with the desired extensions
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()

# Parameters
fps = 30  # Frames per second for the output video
hold_frames = int(2.5 * fps)  # Duration each image is shown (2.5 seconds)
transition_frames = 10  # Total frames for the black flash transition (5 frames fade-out, 5 frames fade-in)

# Function to create a black flash transition between two images
def create_black_flash_transition(img1, img2, num_frames):
    frames = []
    half = num_frames // 2
    # Fade out: from img1 to black
    for i in range(half):
        alpha = i / half
        black_img = np.full_like(img1, 0)
        frame = cv2.addWeighted(img1, 1 - alpha, black_img, alpha, 0)
        frames.append(frame)
    # Fade in: from black to img2
    for i in range(half, num_frames):
        alpha = (i - half) / half
        black_img = np.full_like(img2, 0)
        frame = cv2.addWeighted(black_img, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    return frames

# List to hold all video frames
all_frames = []

# Load the first image and use its dimensions as a reference
if not image_files:
    raise ValueError("No images found in the specified folder.")

# Read the first image and get its size
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
h, w = first_img.shape[:2]

# Process each image
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image cannot be read

    # Resize image if necessary to match the dimensions of the first image
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    # Hold the current image for 2.5 seconds
    for _ in range(hold_frames):
        all_frames.append(img)

    # If this is not the last image, create a black flash transition to the next image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (h, w):
            next_img = cv2.resize(next_img, (w, h))
        transition = create_black_flash_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Save the frames as a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Transition video saved to:", output_video_path)

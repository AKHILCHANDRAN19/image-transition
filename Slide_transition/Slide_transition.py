import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (adjust as needed for your mobile device)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "slide_transition_output.avi")

# Collect image files from the Downloads folder with extensions: .png, .jpg, .jpeg
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()  # sort images alphabetically; adjust if needed

if not image_files:
    raise ValueError("No images found in the specified folder.")

# Video parameters
fps = 30                              # Frames per second for the output video
hold_duration = 2.5                   # Each image is shown for 2.5 seconds
hold_frames = int(hold_duration * fps)  # Number of frames to hold each image
transition_frames = 30                # Number of frames for the slide transition effect

# Read the first image to determine the video dimensions
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
height, width = first_img.shape[:2]

# Function to create a slide transition between two images
def slide_transition(img1, img2, num_frames):
    frames = []
    # Resize both images if needed to ensure they share the same dimensions
    if img1.shape[:2] != (height, width):
        img1 = cv2.resize(img1, (width, height))
    if img2.shape[:2] != (height, width):
        img2 = cv2.resize(img2, (width, height))
    
    for i in range(num_frames):
        # Calculate offset: how many pixels to slide (from 0 to full width)
        offset = int((i / (num_frames - 1)) * width)
        # Create a blank frame (black background)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # For the first image: take the part that is still visible on the right.
        # At offset=0, the full img1 is visible; at offset=width, none of img1 is visible.
        if offset < width:
            frame[:, :width - offset] = img1[:, offset:width]
        
        # For the second image: take the part sliding in from the right.
        # At offset=0, nothing is visible; at offset=width, the full img2 is visible.
        if offset > 0:
            frame[:, width - offset:] = img2[:, :offset]
        
        frames.append(frame)
    return frames

# List to hold all frames of the final video
all_frames = []

# Process each image: hold each image, then apply slide transition to the next image if available
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image cannot be read
    # Resize the image if its dimensions do not match the reference dimensions
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    
    # Hold the current image for the specified duration (2.5 seconds)
    for _ in range(hold_frames):
        all_frames.append(img)
    
    # If not the last image, generate slide transition frames to the next image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (height, width):
            next_img = cv2.resize(next_img, (width, height))
        transition = slide_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Create a VideoWriter object to compile frames into a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write all frames to the video file
for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Slide transition video saved to:", output_video_path)

import cv2
import numpy as np
import os
import glob

# Configuration
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "black_transition_output.avi")
image_extensions = ('*.png', '*.jpg', '*.jpeg')

# Parameters
fps = 30
hold_duration = 2.5  # Image display time
transition_duration = 0.5  # Black transition duration (500ms)

# Collect images
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()

if not image_files:
    raise ValueError("No images found in directory")

# Video setup
first_img = cv2.imread(image_files[0])
h, w = first_img.shape[:2]
hold_frames = int(hold_duration * fps)
transition_frames = int(transition_duration * fps)

def create_black_transition(img1, img2, num_frames):
    frames = []
    half = num_frames // 2
    
    # Fade out to black [[6]]
    for i in range(half):
        alpha = i / half
        black_img = np.full_like(img1, 0)  # Changed from 255 (white) to 0 (black)
        frame = cv2.addWeighted(img1, 1 - alpha, black_img, alpha, 0)
        frames.append(frame)
    
    # Fade in from black [[6]]
    for i in range(half, num_frames):
        alpha = (i - half) / half
        black_img = np.full_like(img2, 0)  # Changed from 255 to 0
        frame = cv2.addWeighted(black_img, 1 - alpha, img2, alpha, 0)
        frames.append(frame)
    
    return frames

# Build frame sequence
all_frames = []
for idx in range(len(image_files)):
    img = cv2.imread(image_files[idx])
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    all_frames.extend([img] * hold_frames)
    
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx+1])
        if next_img.shape[:2] != (h, w):
            next_img = cv2.resize(next_img, (w, h))
            
        transition = create_black_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Export video [[4]]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    out.write(frame.astype(np.uint8))

out.release()
print(f"Black transition video saved to: {output_video_path}")

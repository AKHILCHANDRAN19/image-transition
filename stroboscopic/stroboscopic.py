import cv2
import numpy as np
import os
import glob

# Configuration
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "stroboscopic_transition.avi")
image_extensions = ('*.png', '*.jpg', '*.jpeg')

# Parameters
fps = 30
hold_duration = 2.5  # Image display time (seconds)
transition_duration = 0.3  # Short duration to minimize discomfort [[5]]
transition_frames = int(transition_duration * fps)

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

def create_stroboscopic_transition(img1, img2, num_frames):
    frames = []
    black_img = np.zeros_like(img1)
    
    # First half: Flash between img1 and black [[6]]
    for i in range(num_frames // 2):
        if i % 2 == 0:
            frames.append(img1)
        else:
            frames.append(black_img)
    
    # Second half: Flash between black and img2 [[6]]
    for i in range(num_frames // 2, num_frames):
        if i % 2 == 0:
            frames.append(black_img)
        else:
            frames.append(img2)
    
    # Ensure last frame is img2 [[2]]
    frames[-1] = img2
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
            
        transition = create_stroboscopic_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Export video with warning [[5]]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    out.write(frame.astype(np.uint8))

out.release()
print(f"Stroboscopic transition saved to: {output_video_path}")
print("WARNING: Stroboscopic effects may cause discomfort - use with caution [[5]]")

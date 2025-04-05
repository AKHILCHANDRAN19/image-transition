import cv2
import numpy as np
import os
import glob

# Configuration
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "wave_transition_output.avi")
image_extensions = ('*.png', '*.jpg', '*.jpeg')

# Parameters
fps = 30
hold_duration = 2.5  # seconds per image
transition_duration = 0.5  # seconds per transition
amplitude = 20  # Wave amplitude
wavelength = 50  # Wave wavelength
speed = 2  # Wave speed

# Collect and prepare images
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()

if not image_files:
    raise ValueError("No images found in the specified folder.")

# Video parameters
first_img = cv2.imread(image_files[0])
h, w = first_img.shape[:2]
hold_frames = int(hold_duration * fps)
transition_frames = int(transition_duration * fps)

# Create meshgrid for distortion
x, y = np.meshgrid(np.arange(w), np.arange(h))
x = x.astype(np.float32)
y = y.astype(np.float32)

def create_wave_transition(img1, img2, num_frames):
    frames = []
    for frame in range(num_frames):
        progress = frame / num_frames
        time = frame * speed
        
        # Calculate wave displacement
        dx = amplitude * np.sin(2 * np.pi * (y / wavelength + time / 100))
        
        # Create remap fields
        map_x = x + dx
        map_y = y
        
        # Apply ripple effect
        distorted = cv2.remap(img1, map_x, map_y, cv2.INTER_LINEAR)
        
        # Blend with next image
        alpha = np.clip(progress * 2, 0, 1)
        blended = cv2.addWeighted(distorted, 1 - alpha, img2, alpha, 0)
        
        frames.append(blended)
    return frames

# Prepare all frames
all_frames = []
for idx in range(len(image_files)):
    img = cv2.imread(image_files[idx])
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    # Hold frames
    all_frames.extend([img] * hold_frames)
    
    # Add transition if not last image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx+1])
        if next_img.shape[:2] != (h, w):
            next_img = cv2.resize(next_img, (w, h))
            
        transition = create_wave_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Write video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    out.write(frame.astype(np.uint8))

out.release()
print(f"Video saved to: {output_video_path}")

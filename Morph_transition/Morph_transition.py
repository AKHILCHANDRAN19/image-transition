import cv2
import numpy as np
import os
import glob

# Configuration
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "morph_transition.mp4")
image_extensions = ('*.png', '*.jpg', '*.jpeg')

# Parameters
fps = 30
hold_duration = 2.5  # Image display time
transition_duration = 1.5  # Morph duration [[7]]
pyr_scale = 0.5  # Pyramid scale factor [[6]]
levels = 3  # Number of pyramid layers [[6]]
winsize = 15  # Window size for flow calculation [[6]]
iterations = 3  # Iteration count [[6]]
poly_n = 5  # Neighborhood size [[6]]
poly_sigma = 1.2  # Gaussian standard deviation [[6]]

# Collect images
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()

if len(image_files) < 2:
    raise ValueError("Need at least two images for morph transition")

# Video setup
first_img = cv2.imread(image_files[0])
h, w = first_img.shape[:2]
hold_frames = int(hold_duration * fps)
transition_frames = int(transition_duration * fps)

def create_morph_transition(img1, img2, num_frames):
    # Convert to grayscale for flow calculation [[6]]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow [[6]][[3]]
    flow = cv2.calcOpticalFlowFarneback(
        prev=gray1,
        next=gray2,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0
    )
    
    # Create grid for remapping [[8]]
    h, w = gray1.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    frames = []
    for frame in range(num_frames):
        progress = frame / num_frames
        
        # Warp image using optical flow [[4]]
        remap_x = x + flow[..., 0] * progress
        remap_y = y + flow[..., 1] * progress
        
        warped = cv2.remap(
            img1,
            remap_x,
            remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Blend with target image [[7]]
        alpha = np.clip(progress * 2, 0, 1)
        blended = cv2.addWeighted(warped, 1 - alpha, img2, alpha, 0)
        frames.append(blended)
    
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
            
        transition = create_morph_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Export video (MP4 for CapCut compatibility) [[4]]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    out.write(frame.astype(np.uint8))

out.release()
print(f"Morph transition saved to: {output_video_path}")

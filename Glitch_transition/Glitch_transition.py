import cv2
import numpy as np
import os
import glob

# Configuration
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "glitch_transition_final.mp4")
image_extensions = ('*.png', '*.jpg', '*.jpeg')

# Parameters (Optimized for CapCut compatibility) [[8]][[4]]
fps = 30
hold_duration = 2.5  # 2.5s image display [[1]]
transition_duration = 1.2  # 1.2s glitch effect [[8]]
glitch_intensity = 40  # Increased distortion [[6]]
noise_strength = 80    # Enhanced noise [[4]]
max_channel_shift = 20  # Stronger RGB splits [[7]]

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

def create_glitch_transition(img1, img2, num_frames):
    frames = []
    for frame in range(num_frames):
        progress = frame / num_frames
        
        # Base image with safety checks [[5]]
        glitched = img1.copy()
        for _ in range(np.random.randint(5, 10)):  # More glitch layers
            width = min(np.random.randint(15, 60), w)
            height = min(np.random.randint(8, 30), h)
            x = np.random.randint(0, max(1, w - width))
            y = np.random.randint(0, max(1, h - height))
            
            glitch_block = np.random.randint(
                0, 256, 
                (height, width, 3), 
                dtype=np.uint8
            )
            glitched[y:y+height, x:x+width] = glitch_block
        
        # Noise layer [[5]]
        noise = np.random.randint(
            -noise_strength, 
            noise_strength, 
            (h, w, 3), 
            dtype=np.int16
        )
        glitched = np.clip(glitched.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Channel shifting with safeguards [[7]]
        b, g, r = cv2.split(glitched)
        shift = max(1, int(max_channel_shift * (1 - progress)))
        
        b = np.roll(b, np.random.randint(-shift, shift+1), axis=(0,1))
        g = np.roll(g, np.random.randint(-shift, shift+1), axis=(0,1))
        r = np.roll(r, np.random.randint(-shift, shift+1), axis=(0,1))
        
        glitched = cv2.merge([b, g, r])
        
        # Smooth blending [[8]]
        alpha = np.clip(progress * 1.5, 0, 1)  # Faster transition
        blended = cv2.addWeighted(glitched, 1 - alpha, img2, alpha, 0)
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
            
        transition = create_glitch_transition(img, next_img, transition_frames)
        all_frames.extend(transition)

# Export video (MP4 for CapCut) [[4]]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    out.write(frame.astype(np.uint8))

out.release()
print(f"Final glitch transition saved to: {output_video_path}")

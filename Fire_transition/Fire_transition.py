import cv2
import numpy as np
import os
import glob

# Set the path for your Downloads folder (modify as needed)
downloads_path = "/storage/emulated/0/Download/"
output_video_path = os.path.join(downloads_path, "fire_transition_output.avi")

# Collect image files with the desired extensions
image_extensions = ('*.png', '*.jpg', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(downloads_path, ext)))
image_files.sort()

# Parameters for the video
fps = 30                  # Frames per second
hold_frames = int(2.5 * fps)  # Duration each image is shown (2.5 seconds)
transition_frames = 30    # Number of frames for the fire particle transition

# Parameters for the fire particle system
num_particles = 200       # Number of fire particles

def create_fire_particle_transition(img1, img2, num_frames=transition_frames, num_particles=num_particles):
    """
    Create a transition using a fire particle simulation that spans the full screen.
    The particles blend from a dark burning coal color to a bright ember color.
    """
    frames = []
    h, w = img1.shape[:2]
    
    # Initialize particles at random positions across the full screen
    particles = np.zeros((num_particles, 2), dtype=np.float32)
    particles[:, 0] = np.random.uniform(0, w, num_particles)  # x positions
    particles[:, 1] = np.random.uniform(0, h, num_particles)  # y positions
    
    # Initialize velocities with a slight upward bias and some horizontal movement
    velocities = np.zeros((num_particles, 2), dtype=np.float32)
    velocities[:, 0] = np.random.uniform(-2, 2, num_particles)  # horizontal velocity
    velocities[:, 1] = np.random.uniform(-5, -1, num_particles)   # upward velocity

    # Define the coal and ember colors (BGR format)
    coal_color = np.array([40, 40, 40], dtype=np.uint8)   # Dark, nearly black (burning coal)
    ember_color = np.array([0, 140, 255], dtype=np.uint8)   # Bright ember (orange-red)
    
    for frame_idx in range(num_frames):
        # Create a black background for drawing particles
        particle_frame = np.zeros_like(img1)
        
        # Update particle positions
        particles += velocities
        
        # Optionally, add a slight acceleration or turbulence if desired
        # velocities[:, 1] += 0.1
        
        # Respawn particles that go off-screen (in any direction) randomly across the full screen
        out_of_bounds = (particles[:, 0] < 0) | (particles[:, 0] > w) | (particles[:, 1] < 0) | (particles[:, 1] > h)
        count_off = np.count_nonzero(out_of_bounds)
        if count_off > 0:
            particles[out_of_bounds, 0] = np.random.uniform(0, w, count_off)
            particles[out_of_bounds, 1] = np.random.uniform(0, h, count_off)
            velocities[out_of_bounds, 0] = np.random.uniform(-2, 2, count_off)
            velocities[out_of_bounds, 1] = np.random.uniform(-5, -1, count_off)
        
        # Draw particles with colors that blend from coal_color to ember_color based on their vertical position
        for (x, y) in particles:
            # Intensity based on vertical position (lower particles appear "hotter")
            intensity = np.clip((h - y) / h, 0, 1)
            # Blend between coal and ember colors
            color = (coal_color * (1 - intensity) + ember_color * intensity).astype(np.uint8)
            cv2.circle(particle_frame, (int(x), int(y)), 3, tuple(int(c) for c in color), -1)
        
        # Compute the blending factor for transitioning between img1 and img2
        alpha = frame_idx / num_frames
        transition_base = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        
        # Blend the particle frame with the transition base
        combined = cv2.addWeighted(transition_base, 1, particle_frame, 0.5, 0)
        frames.append(combined)
    
    return frames

# List to hold all video frames
all_frames = []

# Check and load images
if not image_files:
    raise ValueError("No images found in the specified folder.")

# Read the first image and get its dimensions
first_img = cv2.imread(image_files[0])
if first_img is None:
    raise ValueError("Unable to load the first image.")
h, w = first_img.shape[:2]

# Process each image and add transitions
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    if img is None:
        continue  # Skip if the image cannot be read

    # Resize image if necessary to match the dimensions of the first image
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    # Hold the current image for a fixed duration
    for _ in range(hold_frames):
        all_frames.append(img)

    # If not the last image, create a fire particle transition to the next image
    if idx < len(image_files) - 1:
        next_img = cv2.imread(image_files[idx + 1])
        if next_img is None:
            continue
        if next_img.shape[:2] != (h, w):
            next_img = cv2.resize(next_img, (w, h))
        transition = create_fire_particle_transition(img, next_img)
        all_frames.extend(transition)

# Save all frames as a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for frame in all_frames:
    video_writer.write(frame)
video_writer.release()

print("Fire particle transition video saved to:", output_video_path)

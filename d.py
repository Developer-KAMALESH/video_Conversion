import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import subprocess

# Hide the main Tkinter window
Tk().withdraw()

# Ask the user to select a video file
video_path = askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

# If the user cancels the dialog, the video_path will be empty
if not video_path:
    print("No file selected. Exiting.")
    exit()

# Ask the user where to save the cropped video file
output_path = asksaveasfilename(
    title="Save Cropped Video As",
    defaultextension=".mp4",
    filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
)

# If the user cancels the dialog, the output_path will be empty
if not output_path:
    print("No save location selected. Exiting.")
    exit()

# Load YOLO model
model = YOLO('E:\\New folder (2)\\ultralytics\\yolov8n.pt')

# Retrieve class names from the YOLO model
class_names = model.names

# Open the selected video file
cap = cv2.VideoCapture(video_path)

# Check if video capture opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file.")
    exit()

# Frame size
frame_height, frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Fixed 9:16 crop dimensions
crop_width = frame_width
crop_height = int(crop_width * 16 / 9)

# Ensure the crop height fits within the frame height
if crop_height > frame_height:
    crop_height = frame_height
    crop_width = int(crop_height * 9 / 16)

# Initialize previous crop center for smooth transition
prev_center_x, prev_center_y = frame_width // 2, frame_height // 2

# Transition speed for smooth movement
transition_speed = 0.01  # Lower value = slower movement

# Define the codec and create a VideoWriter object to save the cropped video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

# Extract audio using ffmpeg (if available)
audio_path = 'extracted_audio.mp3'
try:
    subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path], check=True)
except FileNotFoundError:
    print("FFmpeg not found, continuing without audio.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    
    # Perform detection
    results = model(frame)

    # Get the detections (bounding boxes, confidence, class labels)
    detections = results[0].boxes

    # Initialize the target center as the previous center (no detection case)
    target_center_x, target_center_y = prev_center_x, prev_center_y
    
    # If there are detections, focus on the largest object (based on area)
    if len(detections) > 0:
        max_area = 0
        for det in detections:
            # Extract bounding box coordinates
            bbox = det.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = bbox[:4]
            
            # Calculate bounding box area
            area = (xmax - xmin) * (ymax - ymin)
            
            # Check if this is the largest detection so far
            if area > max_area:
                max_area = area
                target_center_x = (xmin + xmax) // 2
                target_center_y = (ymin + ymax) // 2

        # Print detected class name
        class_id = int(det.cls)
        class_name = class_names[class_id]
        print(f"Detected: {class_name}, Tracking the object")

    # Smooth transition: Interpolate between previous and target center positions
    center_x = int(prev_center_x + (target_center_x - prev_center_x) * transition_speed)
    center_y = int(prev_center_y + (target_center_y - prev_center_y) * transition_speed)
    
    # Update previous center for the next frame
    prev_center_x, prev_center_y = center_x, center_y
    
    # Calculate crop coordinates
    crop_x1 = center_x - crop_width // 2
    crop_y1 = center_y - crop_height // 2
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    # Ensure crop does not exceed frame boundaries
    if crop_x1 < 0:
        crop_x1 = 0
        crop_x2 = crop_width
    if crop_y1 < 0:
        crop_y1 = 0
        crop_y2 = crop_height
    if crop_x2 > frame_width:
        crop_x2 = frame_width
        crop_x1 = frame_width - crop_width
    if crop_y2 > frame_height:
        crop_y2 = frame_height
        crop_y1 = frame_height - crop_height

    # Crop the frame
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Write the cropped frame to the output video
    out.write(cropped_frame)

    # Display the cropped frame (for debugging)
    cv2.imshow('Cropped Frame', cropped_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Combine audio with video using ffmpeg
try:
    final_output_path = output_path.replace(".mp4", "_with_audio.mp4")
    subprocess.run(['ffmpeg', '-i', output_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_output_path], check=True)
    print(f"Final video saved with audio at: {final_output_path}")
except FileNotFoundError:
    print("FFmpeg not found, audio not combined.")

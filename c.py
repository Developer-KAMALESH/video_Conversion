import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('E:\\New folder (2)\\ultralytics\\yolov8n.pt')

# Get the class names from the model
class_names = model.names

# Open video file or capture from webcam
video_path = 'E:\\New folder (2)\\ultralytics\\MATTA.mp4'  # Update with your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video capture opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file or capture device.")
    exit()

# Frame size
frame_height, frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Fixed 9:16 crop dimensions (based on the video frame width)
crop_width = frame_width  # Width is fixed to the video width
crop_height = int(crop_width * 16 / 9)  # 9:16 aspect ratio

# Ensure the crop height fits within the frame height
if crop_height > frame_height:
    crop_height = frame_height
    crop_width = int(crop_height * 9 / 16)

# Initialize previous crop center for smooth transition
prev_center_x, prev_center_y = frame_width // 2, frame_height // 2

# Transition speed
transition_speed = 0.1  # Adjust this for smoothness (lower value = slower movement)

# Define the codec and create a VideoWriter object to save the cropped video
output_path = 'E:\\New folder (2)\\ultralytics\\portrait\\cropped_out.mp4'  # Output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video format
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frame rate of the original video
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    
    # Perform detection
    results = model(frame)  # Get the model results
    
    # Get the detections (bounding boxes, confidence, class labels)
    detections = results[0].boxes  # Access the bounding boxes from the first result
    
    # Initialize the target center as the previous center (no detection case)
    target_center_x, target_center_y = prev_center_x, prev_center_y
    
    # If there are detections, focus on the largest object (based on area)
    if len(detections) > 0:
        max_area = 0
        for det in detections:
            # Extract bounding box coordinates
            bbox = det.xyxy[0].cpu().numpy()  # Convert to numpy array
            xmin, ymin, xmax, ymax = bbox[:4]
            
            # Calculate bounding box area
            area = (xmax - xmin) * (ymax - ymin)
            
            # Check if this is the largest detection so far
            if area > max_area:
                max_area = area
                # Calculate the center of the largest detected object
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
    
    # Calculate crop coordinates (fixed size centered around the detected object)
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

    # Display the cropped frame
    cv2.imshow('Cropped Frame', cropped_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the video
cv2.destroyAllWindows()

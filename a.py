import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('E:\\New folder (2)\\ultralytics\\yolov8n.pt')

# Get the class names from the model (model.names stores class names)
class_names = model.names

# Open video file or capture from webcam
video_path = 'E:\\New folder (2)\\ultralytics\\MATTA.mp4'  # Update with your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video capture opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file or capture device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    
    # Perform detection
    results = model(frame)  # Get the model results
    
    # Get the detections (bounding boxes, confidence, class labels)
    detections = results[0].boxes  # Access the bounding boxes from the first result
    
    # Iterate over detected objects
    for det in detections:
        # Extract bounding box coordinates
        bbox = det.xyxy[0].cpu().numpy()  # Convert to numpy array
        xmin, ymin, xmax, ymax = bbox[:4]
        
        # Get the class number and map to class name
        class_id = int(det.cls)
        class_name = class_names[class_id]
        
        # Draw bounding box on the frame
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        
        # Put class name above the bounding box
        cv2.putText(frame, f"{class_name}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Print detected class name
        print(f"Detected: {class_name}")
    
    # Display the frame with bounding boxes and class names
    cv2.imshow('Frame', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

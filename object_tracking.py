import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO('yolo11l.pt')  # Replace with your trained model
class_names = model.names

# Video path
video_path = '/kaggle/input/13-12-2024/videoplayback.mp4'  # Update to your dataset folder
cap = cv2.VideoCapture(video_path)

# Output video settings
output_path = '/kaggle/working/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Entry and Exit lines (y-coordinates)
entry_line_y = 100  # Green line
exit_line_y = 700   # Red line

# Initialize counters
entry_count = 0
exit_count = 0

# Track IDs and previous positions
previous_positions = defaultdict(lambda: None)  # {track_id: (x, y)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track objects
    results = model.track(frame, persist=True)
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box
            class_name = class_names[class_idx]

            # Only count cars (adjust class name if needed)
            # Replace the car with the specific class you want 
            if class_name.lower() == "car":
                previous_position = previous_positions[track_id]
                current_position = (cx, cy)

                # Check if the car crosses the entry line (green)
                if previous_position:
                    prev_y = previous_position[1]
                    # Check upward movement across the entry line
                    if prev_y < entry_line_y <= cy:  # Downward crossing
                        entry_count += 1
                        print(f"Car {track_id} crossed the entry line.")

                    # Check downward movement across the exit line
                    elif prev_y < exit_line_y <= cy:  # Upward crossing
                        exit_count += 1
                        print(f"Car {track_id} crossed the exit line.")

                # Update previous position for this track_id
                previous_positions[track_id] = current_position

            # Draw bounding box and center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Red point
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw entry (green) and exit (red) lines
    cv2.line(frame, (0, entry_line_y), (width, entry_line_y), (0, 255, 0), 2)  # Green line
    cv2.line(frame, (0, exit_line_y), (width, exit_line_y), (0, 0, 255), 2)  # Red line

    # Display entry and exit counts
    cv2.putText(frame, f"Entry Count: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit Count: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()

# Display the output video path
print(f"Processed video saved at: {output_path}")

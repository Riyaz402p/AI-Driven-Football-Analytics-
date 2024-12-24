import cv2
from ultralytics import YOLO
model = YOLO('yolo11l.pt')
class_names = model.names
import cv2
import numpy as np

def get_center_frame(video_path):
    """Capture the center frame of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read frame.")
        return None

    return frame

def draw_polygon(event, x, y, flags, param):
    """Mouse callback function to draw polygons."""
    global points, drawing, polygon_count

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        if len(points) == polygon_sides:
            polygons.append(points.copy())
            print(f"Polygon {len(polygons)} coordinates: {points}")
            points = []
            polygon_count -= 1

            if polygon_count == 0:
                print("Finished drawing all polygons.")

def main():
    global points, drawing, polygon_count, polygon_sides, polygons

    video_path = input("Enter the path to the video file: ")
    frame = get_center_frame(video_path)

    if frame is None:
        return

    polygon_sides = int(input("Enter the number of sides for the polygon: "))
    polygon_count = int(input("How many polygons do you want to draw? "))

    points = []
    polygons = []

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_polygon)

    while True:
        display_frame = frame.copy()

        # Draw the polygons on the frame
        for polygon in polygons:
            cv2.polylines(display_frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the current in-progress polygon
        if len(points) > 1:
            cv2.polylines(display_frame, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=1)

        # Display the coordinates on the frame
        for i, polygon in enumerate(polygons):
            coordinates_text = f"Polygon {i + 1}: {polygon}"
            cv2.putText(display_frame, coordinates_text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Frame", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

        if polygon_count == 0:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
cap = cv2.VideoCapture(r'C:\Users\DELL\Downloads\istockphoto-1492560725-640_adpp_is.mp4')
import cv2
import time
from collections import defaultdict

# Polygon coordinates
polygons = {
    1: [(646, 170), (694, 182), (566, 428), (291, 430)],  # Polygon 1
    2: [(335, 215), (394, 212), (5, 391), (6, 338)],      # Polygon 2
    3: [(302, 171), (321, 182), (1, 269), (2, 224)],      # Polygon 3
    4: [(335, 191), (371, 179), (563, 230), (523, 261)],  # Polygon 4
    5: [(671, 140), (715, 143), (693, 182), (648, 168)],  # Polygon 5
    6: [(503, 194), (459, 174), (500, 152), (505, 157)],  # Polygon 6
    7: [(343, 175), (365, 178), (376, 156), (363, 142)],  # Polygon 7
    8: [(308, 153), (237, 172), (247, 181), (311, 168)],  # Polygon 8
}

# Time tracking for people inside each polygon
person_time_in_polygon = {i: defaultdict(int) for i in polygons}

# Variables to track total people and unique IDs
unique_track_ids = set()
overall_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes = 0 )  

    if results[0].boxes.data is not None:
        # Get detected boxes, class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []  # Handle as needed
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw the polygons on the frame
        for idx, polygon in polygons.items():
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 2)
            cv2.putText(frame, f"Polygon {idx}", polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_names[class_idx]

            # Update unique track IDs
            if track_id not in unique_track_ids:
                unique_track_ids.add(track_id)
                overall_count += 1  # Increment overall count for new objects

            # Check if the person is inside any of the polygons and update the time spent
            for polygon_id, polygon in polygons.items():
                if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                    person_time_in_polygon[polygon_id][track_id] += 1  # Increment time spent in the polygon

            # Draw center point and bounding box
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display counts for each polygon
        for polygon_id, person_times in person_time_in_polygon.items():
            cv2.putText(frame, f"Polygon {polygon_id} - People: {len(person_times)}", 
                        (10, 30 * polygon_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the time each person spent inside each polygon
        for polygon_id, person_times in person_time_in_polygon.items():
            max_time_id = max(person_times, key=person_times.get, default=None)
            y_offset = 50 + 30 * polygon_id
            for track_id, time_spent in person_times.items():
                minutes_spent = time_spent / 30  # Convert frames to minutes (assuming 30 FPS)
                color = (255, 255, 0) if track_id != max_time_id else (0, 0, 255)  # Highlight max time person
                cv2.putText(frame, f"Polygon {polygon_id} - ID {track_id} Time: {minutes_spent:.2f} min", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20

        # Display the frame with tracking
        cv2.imshow('yolo_tracking', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
import numpy as np
from collections import defaultdict

# Polygon coordinates
polygons = {
    1: [(646, 170), (694, 182), (566, 428), (291, 430)],  # Polygon 1
    2: [(335, 215), (394, 212), (5, 391), (6, 338)],      # Polygon 2
    3: [(302, 171), (321, 182), (1, 269), (2, 224)],      # Polygon 3
    4: [(335, 191), (371, 179), (563, 230), (523, 261)],  # Polygon 4
    5: [(671, 140), (715, 143), (693, 182), (648, 168)],  # Polygon 5
    6: [(503, 194), (459, 174), (500, 152), (505, 157)],  # Polygon 6
    7: [(343, 175), (365, 178), (376, 156), (363, 142)],  # Polygon 7
    8: [(308, 153), (237, 172), (247, 181), (311, 168)],  # Polygon 8
}

# Time tracking for people inside each polygon
person_time_in_polygon = {i: defaultdict(int) for i in polygons}

# Variables to track total people and unique IDs
unique_track_ids = set()
overall_count = 0

# Store complete track data
track_data = defaultdict(list)

frame_count = 0  # Initialize frame_count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Increment the frame count

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=0)

    if results[0].boxes.data is not None:
        # Get detected boxes, class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []  # Handle as needed
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw the polygons on the frame
        for idx, polygon in polygons.items():
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 2)
            cv2.putText(frame, f"Polygon {idx}", polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_names[class_idx]

            # Update unique track IDs
            if track_id not in unique_track_ids:
                unique_track_ids.add(track_id)
                overall_count += 1  # Increment overall count for new objects

            # Store complete coordinates frame-wise
            track_data[track_id].append({
                'frame': frame_count,
                'bounding_box': (x1, y1, x2, y2),
                'center': (cx, cy)
            })

            # Check if the person is inside any of the polygons and update the time spent
            for polygon_id, polygon in polygons.items():
                if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                    person_time_in_polygon[polygon_id][track_id] += 1  # Increment time spent in the polygon

            # Draw center point and bounding box
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display counts for each polygon
        for polygon_id, person_times in person_time_in_polygon.items():
            cv2.putText(frame, f"Polygon {polygon_id} - People: {len(person_times)}",
                        (10, 30 * polygon_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame with tracking
        cv2.imshow('yolo_tracking', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save or print track data
for track_id, frames in track_data.items():
    print(f"Track ID {track_id}:")
    for data in frames:
        print(f"  Frame {data['frame']} - Bounding Box: {data['bounding_box']} - Center: {data['center']}")
import cv2
import numpy as np
import csv
from collections import defaultdict
from ultralytics import YOLO

# Polygon coordinates
polygons = {
    1: [(646, 170), (694, 182), (566, 428), (291, 430)],  # Polygon 1
    2: [(335, 215), (394, 212), (5, 391), (6, 338)],      # Polygon 2
    3: [(302, 171), (321, 182), (1, 269), (2, 224)],      # Polygon 3
    4: [(335, 191), (371, 179), (563, 230), (523, 261)],  # Polygon 4
    5: [(671, 140), (715, 143), (693, 182), (648, 168)],  # Polygon 5
    6: [(503, 194), (459, 174), (500, 152), (505, 157)],  # Polygon 6
    7: [(343, 175), (365, 178), (376, 156), (363, 142)],  # Polygon 7
    8: [(308, 153), (237, 172), (247, 181), (311, 168)],  # Polygon 8
}


# Variables for tracking
person_time_in_polygon = {i: defaultdict(int) for i in polygons}
unique_track_ids = set()
frame_count = 0

# Initialize video capture (replace with your video source)
#cap = cv2.VideoCapture("your_video.mp4")
cap = cv2.VideoCapture(r'C:\Users\DELL\Downloads\istockphoto-1492560725-640_adpp_is.mp4')
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()


model = YOLO('yolo11l.pt')
class_names = ["person"]  # Replace with actual class names

# CSV file setup
csv_file = "track_data.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "Frame", "Center X", "Center Y"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment frame count

        # Run YOLO tracking on the frame (replace with actual model logic)
        results = model.track(frame, persist=True, classes=0)  # Replace with your YOLO detection function

        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if track_id not in unique_track_ids:
                    unique_track_ids.add(track_id)

                # Write to CSV
                writer.writerow([track_id, frame_count, cx, cy])

                # Check polygons
                for polygon_id, polygon in polygons.items():
                    if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                        person_time_in_polygon[polygon_id][track_id] += 1

cap.release()
print(f"Tracking data saved to {csv_file}")
import csv
import numpy as np
from collections import defaultdict

# Polygon coordinates
polygons = {
    1: [(646, 170), (694, 182), (566, 428), (291, 430)],  # Polygon 1
    2: [(335, 215), (394, 212), (5, 391), (6, 338)],      # Polygon 2
    3: [(302, 171), (321, 182), (1, 269), (2, 224)],      # Polygon 3
    4: [(335, 191), (371, 179), (563, 230), (523, 261)],  # Polygon 4
    5: [(671, 140), (715, 143), (693, 182), (648, 168)],  # Polygon 5
    6: [(503, 194), (459, 174), (500, 152), (505, 157)],  # Polygon 6
    7: [(343, 175), (365, 178), (376, 156), (363, 142)],  # Polygon 7
    8: [(308, 153), (237, 172), (247, 181), (311, 168)],  # Polygon 8
}
# Define video properties (replace with your actual frame rate)
fps = 0.1
time_per_frame = 1 / fps

# Variables for analysis
track_data = defaultdict(list)
time_spent_in_polygons = defaultdict(lambda: defaultdict(float))

# Read CSV
csv_file = "track_data.csv"
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_id = int(row["Track ID"])
        frame = int(row["Frame"])
        cx, cy = int(row["Center X"]), int(row["Center Y"])
        track_data[track_id].append({"frame": frame, "center": (cx, cy)})

        for polygon_id, polygon in polygons.items():
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                time_spent_in_polygons[polygon_id][track_id] += time_per_frame

# Analyze peak times
frame_count = max(data['frame'] for frames in track_data.values() for data in frames)
peak_time_in_polygons = defaultdict(int)

for frame in range(1, frame_count + 1):
    people_in_polygons = defaultdict(int)
    for track_id, frames in track_data.items():
        for data in frames:
            if data["frame"] == frame:
                cx, cy = data["center"]
                for polygon_id, polygon in polygons.items():
                    if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                        people_in_polygons[polygon_id] += 1
    for polygon_id, count in people_in_polygons.items():
        peak_time_in_polygons[polygon_id] = max(peak_time_in_polygons[polygon_id], count)

# Generate report
print("Time Spent in Polygons:")
for polygon_id, track_times in time_spent_in_polygons.items():
    max_time_track = max(track_times, key=track_times.get)
    max_time = track_times[max_time_track]
    print(f"Polygon {polygon_id}:")
    print(f"  Maximum time spent: {max_time:.2f} seconds by Track ID {max_time_track}")
    print(f"  Total unique visitors: {len(track_times)}")
    for track_id, time_spent in track_times.items():
        print(f"    Track ID {track_id}: {time_spent:.2f} seconds")

print("\nPeak Time Analysis:")
for polygon_id, peak_count in peak_time_in_polygons.items():
    print(f"Polygon {polygon_id} had a peak of {peak_count} people simultaneously.")

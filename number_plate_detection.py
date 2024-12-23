import cv2
from ultralytics import YOLO
import easyocr

# Function to crop detected number plate region
def crop_number_plate(image, bbox):
    x1, y1, x2, y2 = map(int, bbox[:4])  # Convert coordinates to integers
    cropped = image[y1:y2, x1:x2]  # Crop using bounding box
    return cropped

# Function to extract text using EasyOCR
def extract_text_with_easyocr(cropped_image, reader):
    results = reader.readtext(cropped_image)
    text = " ".join([result[1] for result in results])  # Extract detected text
    return text.strip()

# Function to process image with YOLO and OCR
def process_image_with_ocr(image_path, model_path):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Run inference
    results = model(image)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            
            # Draw bounding box on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop the detected number plate
            cropped_plate = crop_number_plate(image, (x1, y1, x2, y2))
            
            # Extract text using EasyOCR
            text = extract_text_with_easyocr(cropped_plate, reader)
            print(f'Detected Text: {text}')
            
            # Overlay detected text on the image
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show cropped plate (optional)
            cv2.imshow("Cropped Number Plate", cropped_plate)
            cv2.waitKey(0)
    
    # Display the image with bounding boxes and text
    cv2.imshow("Detected Number Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
image_path = r'G:\yolo11Project\Testimage\images (2).jpg'
model_path = r'G:\yolo11Project\detect\train\weights\best.pt'

process_image_with_ocr(image_path, model_path)

import cv2
import easyocr
import re
import os
from datetime import datetime

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # For English - 'en'

# Recognize text function
def recognize_text(frame):
    '''Recognizes text from a video frame using EasyOCR.'''
    return reader.readtext(frame)

# Function to extract MRP and expiry date using regex
def extract_product_details(extracted_text):
    '''Parses the extracted text to find MRP and expiry date.'''
    
    # Initialize a dictionary to store detected product details
    product_details = {
        "MRP": None,
        "Date of Expiry": None
    }

    # Extract MRP
    mrp_match = re.search(r'MRP\s*[:\s]([\d,.]+)\s(Rs|₹|/-)', extracted_text, re.IGNORECASE)
    if mrp_match:
        product_details["MRP"] = f"₹ {mrp_match.group(1)}"  # Format as currency

    # Extract Date of Expiry with improved regex
    expiry_date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', extracted_text)
    if expiry_date_match:
        product_details["Date of Expiry"] = expiry_date_match.group(1)

    return product_details

# Overlay function to add detected text and bounding boxes
def overlay_ocr_text(frame, prob_threshold=0.5, rect_thickness=2, font_scale=0.8):
    '''Recognizes text and overlays it on the video frame with bounding boxes.'''
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Increase contrast
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    contrasted = cv2.convertScaleAbs(adaptive_thresh, alpha=alpha, beta=beta)

    # Sharpen the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.filter2D(contrasted, -1, kernel)

    # Recognize text using the processed frame
    result = recognize_text(sharpened)
    extracted_texts = []

    # Loop through recognized texts
    for (bbox, text, prob) in result:
        if prob >= prob_threshold:  # Use threshold to filter low-probability detections
            extracted_texts.append(text)
            print(f'Detected text: {text} (Confidence: {prob:.2f})')

            # Extract bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # Draw bounding box around detected text
            cv2.rectangle(frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=rect_thickness)

            # Put the recognized text above the rectangle
            cv2.putText(frame, text, org=(top_left[0], top_left[1] - 10), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 255, 0), thickness=2)

    # Join all the extracted texts into a single block of text
    combined_text = ' '.join(extracted_texts)
    print(f"Extracted Text: {combined_text}")  # Debugging line

    # Extract product details
    product_details = extract_product_details(combined_text)
    print("\n--- Detected Product Details ---")
    for key, value in product_details.items():
        print(f"{key}: {value if value else 'Not Found'}")

    return frame

# Function to save captured images with dynamic filenames
def save_image(frame, directory='captured_images'):
    '''Saves captured frames with unique filenames in a specified directory.'''
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(directory, f"captured_image_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    print(f"Image captured and saved as '{file_path}'.")

# Start video capture from the default camera (change to 1 for external cameras)
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open video. Please check your camera.")
else:
    print("Press 'c' to capture an image, 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the camera feed using OpenCV
        cv2.imshow('Camera Feed', frame)

        # Capture image on 'c' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Process the captured frame with text overlay
            processed_frame = overlay_ocr_text(frame)

            # Display the processed image using OpenCV
            cv2.imshow('Processed Image', processed_frame)

            # Save the processed frame
            save_image(processed_frame)

        elif key == ord('q'):
            break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
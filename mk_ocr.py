import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import numpy as np
import spacy
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Path to the passport image
image_path = './Kyrgyzstan.jpg'

# Open the image with PIL
pil_image = Image.open(image_path)

# Convert PIL image to OpenCV format (NumPy array)
opencv_image = np.array(pil_image)

# Ensure image format is compatible with OpenCV
if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 4:
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGR)
elif len(opencv_image.shape) == 2:  # Handle grayscale images
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
elif len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
else:
    raise ValueError("Unsupported image format")

# Preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    return image

# Preprocess the image before OCR
preprocessed_image = preprocess_image(opencv_image)

# Extract text
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang='eng') 

# Process text with SpaCy's NER pipeline
doc = nlp(text)

# Extract specific fields from text
def extract_fields(text):
    fields = {
        "Name": None,
        "Date of Birth": None,
        "Date of Issue": None,
        "Date of Expiry": None,
        "Authority": None,
        "Nationality": None,
        "Passport Type": None,
        "Passport Number": None
    }
    
    # Define patterns for extracting dates
    date_pattern = r'\d{2} [A-Z][a-z]{2} \d{4}'
    
    # Extract specific fields using regex or keyword search
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        if "Name" in line:
            fields["Name"] = line.split("Name")[1].strip()
        elif "Date of Birth" in line:
            fields["Date of Birth"] = re.search(date_pattern, line).group() if re.search(date_pattern, line) else None
        elif "Date of Issue" in line:
            fields["Date of Issue"] = re.search(date_pattern, line).group() if re.search(date_pattern, line) else None
        elif "Date of Expiry" in line:
            fields["Date of Expiry"] = re.search(date_pattern, line).group() if re.search(date_pattern, line) else None
        elif "Authority" in line:
            fields["Authority"] = line.split("Authority")[1].strip()
        elif "Nationality" in line:
            fields["Nationality"] = line.split("Nationality")[1].strip()
        elif "Passport Type" in line:
            fields["Passport Type"] = line.split("Passport Type")[1].strip()
        elif "Passport Number" in line:
            fields["Passport Number"] = line.split("Passport Number")[1].strip()
    
    return fields

# Extract fields from the text
fields = extract_fields(text)

# Print the extracted fields
print("Extracted Fields:")
for key, value in fields.items():
    print(f"{key}: {value}")

# Draw bounding boxes on the image (if needed)
def draw_bounding_boxes(image, detection_boxes, entities):
    height, width, _ = image.shape
    
    for i in range(len(detection_boxes['char'])):
        left = detection_boxes['left'][i]
        top = detection_boxes['top'][i]
        right = detection_boxes['right'][i]
        bottom = detection_boxes['bottom'][i]
        
        # Draw rectangle in green color
        cv2.rectangle(image, (left, height - bottom), (right, height - top), (0, 255, 0), 2)

    for ent in entities:
        if ent.label_ in ["PERSON", "GPE", "ORG", "DATE"]:
            start = ent.start_char
            end = ent.end_char
            if start >= 0 and end >= 0:
                cv2.rectangle(image, (start, height - end), (end, height - start), (255, 0, 0), 2)

    return image

# Draw bounding boxes on the OpenCV image
image_with_boxes = draw_bounding_boxes(opencv_image, pytesseract.image_to_boxes(preprocessed_image, output_type=Output.DICT), doc.ents)

# Convert OpenCV image back to PIL format for saving
image_with_boxes_pil = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))

# Save the image with bounding boxes
image_with_boxes_pil.save('passport_with_boxes.jpg')

print("Bounding boxes have been drawn and image saved.")

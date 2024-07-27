import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import numpy as np
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Path to the passport image
image_path = './Iceland.jpg'

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
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    # Optional: Resize image if needed
    resized = cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
    return denoised

# Preprocess the image before OCR
preprocessed_image = preprocess_image(opencv_image)

# Perform OCR to get bounding boxes
detection_boxes = pytesseract.image_to_boxes(preprocessed_image, output_type=Output.DICT)
    
# Extract text
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang='eng') 
# text = pytesseract.image_to_string(preprocessed_image)

# Process text with SpaCy's NER pipeline
doc = nlp(text)

# Draw bounding boxes on the image
def draw_bounding_boxes(image, detection_boxes, entities):
    height, width, _ = image.shape
    
    for i in range(len(detection_boxes['char'])):
        left = detection_boxes['left'][i]
        top = detection_boxes['top'][i]
        right = detection_boxes['right'][i]
        bottom = detection_boxes['bottom'][i]
        
        # Draw rectangle in green color
        cv2.rectangle(image, (left, height - bottom), (right, height - top), (0, 255, 0), 2)

    return image

def named_entity_recognition():
    info = {
        "PERSON": [],
        "DATE": [],
        "ORG": [],
        "GPE": [],
    }
    
    for ent in doc.ents:
        if ent.label_ in info:
            info[ent.label_].append(ent.text)
    
    # Print recognized entities
    print("Organizations (ORG):", info["ORG"])
    print("Names (PERSON):", info["PERSON"])
    print("Dates (DATE):", info["DATE"])
    print("Geopolitical Entities (GPE):", info["GPE"])
    
    return info

# Extract and print named entities
entities_info = named_entity_recognition()

# Draw bounding boxes on the OpenCV image
image_with_boxes = draw_bounding_boxes(opencv_image, detection_boxes, doc.ents)

# Convert OpenCV image back to PIL format for saving
image_with_boxes_pil = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))

# Save the image with bounding boxes
image_with_boxes_pil.save('passport_with_boxes.jpg')

print("Bounding boxes have been drawn and image saved.")

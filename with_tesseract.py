import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
import re
import spacy
import numpy as np


def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply thresholding
    # _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # denoising
    # binary_image = cv2.medianBlur(binary_image, 3)
    # resizing
    binary_image = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Save preprocessed image
    preprocessed_image_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_image_path, binary_image)
    return preprocessed_image_path

def get_dates(text):
    dates = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
    if not dates:
        dates = re.findall(r'\b\d{2}-\d{2}-\d{4}\b', text)
    if not dates:
        dates = re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', text)
    if not dates:
        dates = re.findall(r'\b\d{2}\s[a-zA-Z]+\s\d{4}\b', text)
    return dates

def named_entity_recognition(text):
    nlp = spacy.load("en_core_web_lg")
    # Process the text
    doc = nlp(text) 
    # Extract entities
    def extract_information(doc):
        info = {
            "PERSON": [],
            "DATE": [],
            "ORG": [],
            "GPE": [],
        }
        
        for ent in doc.ents:
            if ent.label_ in info:
                info[ent.label_].append(ent.text)
        
        return info
    
    # Extract and print information
    info = extract_information(doc)
    # # print all entities
    # print(info)
    print("Name:", info["ORG"])
    print("Misc:", info["PERSON"])
    print("Dates (DATE):", info["DATE"])
    print("Geopolitical Entities (GPE):", info["GPE"])
    
    return info

def ocr_with_tesseract(image, lang='eng'):
    # preprocessed_image_path = preprocess_image(image_path)
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config, lang=lang) 
    dates = get_dates(text)
    return dates, text

def bounding_boxes(image_path):
    # Open the image with PIL
    pil_image = Image.open(image_path)

    # Convert PIL image to OpenCV format
    opencv_image = np.array(pil_image)

    # Check if the conversion worked and ensure it's in the correct format
    if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 4:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGR)
    elif len(opencv_image.shape) == 2:  # Handle grayscale images
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
    elif len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image format")

    # Perform OCR to get bounding boxes
    detection_boxes = pytesseract.image_to_boxes(opencv_image, output_type=Output.DICT)
    height, width, _ = opencv_image.shape
    
    # time to draw!
    for i in range(len(detection_boxes['char'])):
        left = detection_boxes['left'][i]
        top = detection_boxes['top'][i]
        right = detection_boxes['right'][i]
        bottom = detection_boxes['bottom'][i]
        
        # Draw rectangle in green color
        cv2.rectangle(image, (left, height - bottom), (right, height - top), (0, 255, 0), 2)
    
    return image



image_path = "./Indonesia.jpg"
image = Image.open(image_path)

dates, text = ocr_with_tesseract(image) # ocr with tesseract, an open sourced model

ner = named_entity_recognition(text) # named entity recognition to get entities from the raw text
bounding_boxes_image = bounding_boxes(image_path) # show bounding boxes around the text in the image 

print(ner)
# print(text)
if(ner['DATE'] == []):
    print(dates) # if spacy model fails in getting any dates, regex will help

# Display the image with bounding boxes
bounding_boxes_image.save('passport_with_boxes.jpg')
print('Image with bounding boxes saved as passport_with_boxes.jpg')

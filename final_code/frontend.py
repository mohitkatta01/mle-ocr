import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import numpy as np
import spacy
import streamlit as st

def process(uploaded_image):
    pil_image = Image.open(uploaded_image)
    opencv_image = np.array(pil_image) # convert to Numpy arrays

    # ensure robustness of image format
    # Ensure image format is compatible with OpenCV
    if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 4:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGR)
    elif len(opencv_image.shape) == 2:  # Handle grayscale images
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
    elif len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image format")
    
    return opencv_image

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    return denoised

def named_entity_recognition(doc):
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

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

def main():
    st.title("Passport OCR and NER Application")
    st.write("This application extracts text from a passport image and performs named entity recognition (NER) on the extracted text.")

    uploaded_image = st.file_uploader("Choose a passport image...", type=["jpg", "jpeg", "png"])

    with st.spinner("Loading SpaCy model..."):
        # Load SpaCy model
        nlp = spacy.load("en_core_web_lg")
        st.toast("SpaCy model loaded successfully",icon="✅")

    if uploaded_image is not None:
        processed_image = process(uploaded_image) # inport image and convert to necessary format
        preprocessed_image = preprocess_image(processed_image) # modify the image for the OCR system to work

        # Perform OCR to get bounding boxes
        detection_boxes = pytesseract.image_to_boxes(preprocessed_image, output_type=Output.DICT)

        # BRAIN of the algorithm : Extract the text
        custom_config = r'--oem 1 --psm 6'
        text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang='eng')
        
        # Process text with SpaCy's NER pipeline
        doc = nlp(text)

        # add all the named entities to a table
        ner = named_entity_recognition(doc)
        st.write("Named Entities:")
        st.table(ner)

        # Draw bounding boxes on the OpenCV image
        image_with_boxes = draw_bounding_boxes(processed_image, detection_boxes, doc.ents)
        st.image(image_with_boxes, caption="Passport image with bounding boxes", use_column_width=True)
        
if __name__ == "__main__":
    main()

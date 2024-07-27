## Passport OCR and NER Application

### Overview

This application extracts text from a passport image and performs named entity recognition (NER) on the extracted text. The application uses Tesseract for Optical Character Recognition (OCR) and SpaCy for Named Entity Recognition (NER). Additionally, the application provides an interface for users to input a name and a date, which are then compared to the recognized entities to calculate accuracy.

### Prerequisites

- Python 3.7 or higher
- Tesseract-OCR installed on your machine
- Internet connection to download SpaCy models

### Installation Steps

1. **Install Tesseract-OCR:**

   - **Windows:**
     Download the Tesseract installer from [here](https://github.com/UB-Mannheim/tesseract/wiki) and follow the installation instructions.

   - **Mac:**
     ```sh
     brew install tesseract
     ```

   - **Linux:**
     ```sh
     sudo apt-get install tesseract-ocr
     ```

2. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
    ```
3. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
4. **Download all required dependecies**
    Libraries Used:

    * *Streamlit*: For creating a web-based user interface.
    * *OpenCV*: For image processing.
    * *Pillow*: For handling image file operations.
    * *NumPy*: For efficient numerical operations on images.
    * *Pandas*: For handling data frames to display NER results.
    * *Tesseract-OCR*: For text extraction from images.
    * *SpaCy*: For named entity recognition.
    * *difflib*: For calculating similarity between strings.

5. **Download SpaCy model**
    ```sh
    python -m spacy download en_core_web_lg
    ```

### Running the application
1. Run the streamlit application
    ```sh
    streamlit run app.py
    ```
2. Upload a passport image
    * Click on "Browse files" to upload a passport image in jpg, jpeg, or png format. (Drag and Drop is also possible)

3. Enter details for accuracy testing:
    * Enter the name and date you want to test for accuracy in the provided text input boxes.

4. View results:
    * The application will display the extracted text, bounding boxes around text segments, and accuracy of the entered name and date.

### Approach

**Preprocessing Steps**

1. Image Conversion:
    * The uploaded image is converted to a NumPy array.
    * Ensures robustness by converting different image formats to BGR format compatible with OpenCV.

2. Grayscale Conversion:
    * The image is converted to grayscale to simplify the OCR process.

3. Thresholding:
    * Apply Otsu's thresholding to binarize the image, which helps in distinguishing text from the background.

4. Denoising:
    * FastNlMeansDenoising is applied to reduce noise and improve text detection.

**OCR (Optical Character Recognition)**

* Tesseract-OCR
    * Tesseract is used to extract text from the preprocessed image.
    * Bounding boxes for detected characters are obtained using `pytesseract.image_to_boxes`.

**NER (Named Entity Recognition)**
* SpaCy
    * SpaCy's large English model (en_core_web_lg) is used for NER.
    * The extracted text is processed to identify entities such as PERSON, ORG, and DATE.

**Accuracy Calculation**
* Text Input:
    * Users can input a name and a date to test the accuracy of the recognized entities.
    * This model uses name of the person and checks with the entities extracted by the NER model

* Similarity Calculation:
    * When there is not a perfect match, The `difflib.SequenceMatcher` is used to calculate the similarity between the input text and recognized entities.

### Dependencies
```
streamlit==1.18.0
opencv-python-headless==4.8.0.74
Pillow==9.2.0
numpy==1.22.4
pandas==1.4.2
pytesseract==0.3.9
spacy==3.4.1
difflib
```

This application provides a comprehensive solution for extracting and analyzing text from passport images, with an easy-to-use interface for validating recognized entities.

### Further Work:
Due to time contraints and limited availability of passport related images, I could not have created a fine-tuned model which extracts data in a much better way. I have currently used the Tesseract model which was pretrained using an LSTM model, configured by Google. There are no pre trainings done by me directly due to the lack of GPU resources and funding.

If time and funds exist, employing GCP/AWS clusters/ Digital Ocean droplets would provide more useful for tasks such as extracting text from passport related documents. The current model is perfect for documents without a lot of color noise and gives a slightly lower accuracy for the task at hand.

Just a note to keep in mind. If you need any help in setting up the system, kindly email/call me so I can walk you through the program step-by-step.

Thank you,
Mohit Katta






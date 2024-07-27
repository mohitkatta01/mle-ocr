import boto3

def ocr_with_aws_textract(image_path):
    client = boto3.client('textract', region_name='us-east-1')
    with open(image_path, 'rb') as document:
        imageBytes = bytearray(document.read())
    response = client.detect_document_text(Document={'Bytes': imageBytes})
    text = ""
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + "\n"
    return text

text = ocr_with_aws_textract('./India.jpg')
print(text)

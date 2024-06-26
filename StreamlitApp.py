import streamlit as st
from pathlib import Path
from google.cloud import vision
import io
import cv2
import base64
from ultralytics import YOLO
import os
import json

def detect_text(image_path):
    """Detect text in the image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    return texts

def draw_annotations(image_path, texts):
    """Draw bounding boxes around detected text and save the annotated image."""
    image = cv2.imread(image_path)
    
    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        for i in range(len(vertices)):
            start_point = vertices[i]
            end_point = vertices[(i + 1) % len(vertices)]
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)
    return image

def save_segmented_objects(results, original_image, output_dir):
    """Save segmented objects from the prediction results."""
    masks = results.masks.data
    boxes = results.boxes.data
    seg_classes = list(results.names.values())
    object_images = []

    for i, mask in enumerate(masks):
        mask_np = mask.cpu().numpy().astype('uint8') * 255
        mask_resized = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))

        if mask_resized.any():
            cls = int(boxes[i, 5])
            cls_name = seg_classes[cls]
            original_image_masked = cv2.bitwise_and(original_image, original_image, mask=mask_resized)
            image_path = str(output_dir / f'{cls_name}_{i}.jpg')
            cv2.imwrite(image_path, original_image_masked)
            
            # Convert image to base64
            with open(image_path, "rb") as img_file:
                object_images.append(base64.b64encode(img_file.read()).decode('utf-8'))

    return object_images

def generate_html(texts, object_images):
    """Generate HTML content to display extracted text and images of visual elements."""
    html_content = '<html><body>'
    
    for object_image in object_images:
        html_content += f'<img src="data:image/jpeg;base64,{object_image}" alt="Object Image" style="width:200px;"><br>'
    
    if texts:
        full_text = texts[0].description
        paragraphs = full_text.split('\n')
        for para in paragraphs:
            if para.strip():
                html_content += f'<p>{para}</p>'
    
    html_content += '</body></html>'
    return html_content

def main():
    st.title("Image Text Extraction")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        image_path = 'uploaded_image.jpg'
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        texts = detect_text(image_path)
        if texts:
            st.header("Text Detected:")
            st.write(texts[0].description)


        model = YOLO("yolov8m-seg.pt")

        
        results = model.predict(image_path)

      
        results[0].save("/tmp/output.jpg")
        saved_image_path = "/tmp/output.jpg"
      
        annotated_image = draw_annotations(saved_image_path, texts)
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Annotated Image', use_column_width=True)
 

        output_dir = Path("./test_output/")
        output_dir.mkdir(parents=True, exist_ok=True)

      
        original_image = cv2.imread(image_path)

       
        object_images = save_segmented_objects(results[0], original_image, output_dir)


        html_content = generate_html(texts, object_images)

    
        with open(output_dir / 'result.html', 'w') as f:
            f.write(html_content)

        st.success("HTML file generated successfully!")
        st.download_button(label="Download HTML File", data=html_content, file_name='result.html')

if __name__ == "__main__":
    
    credentials_content = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]['credentials']
    credentials_path = "/tmp/credentials.json"
    with open(credentials_path, 'w') as f:
        json.dump(dict(credentials_content), f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    main()

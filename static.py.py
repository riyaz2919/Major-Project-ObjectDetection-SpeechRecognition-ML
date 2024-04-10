import cv2
import numpy as np
import pyttsx3
import base64
import tempfile
import os
import threading
from flask import Flask, request, render_template, jsonify

import google.generativeai as genai

classNames = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

proto = "D:\\majorProject\\SSD_using_OpenCV\\MobileNetSSD_deploy.prototxt"
weights = "D:\\majorProject\\SSD_using_OpenCV\\MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, weights)

GOOGLE_API_KEY ='AIzaSyCa3y7CsQIemJkig9USovhYn0U8vZ3tUu4'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

# Create a directory for storing temporary files
temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
os.makedirs(temp_dir, exist_ok=True)

def detect_objects(image_path):
    with open(image_path, 'rb') as file:
        img_data = file.read()
    
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    img_resized = cv2.resize(img, (300, 300))

    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300),
                                 (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    height, width, _ = img.shape

    final = detections.squeeze()

    objects_detected = []

    for i in range(final.shape[0]):
        conf = final[i, 2]
        if conf > 0.5:
            class_name = classNames[final[i, 1]]
            objects_detected.append(class_name)
            x1, y1, x2, y2 = final[i, 3:]
            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height
            top_left = (int(x1), int(y1))
            bottom_right = (int(x2), int(y2))
            img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
            img = cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    tts_text = "Detected objects: " + ", ".join(objects_detected)
    print(tts_text)

    # Start a new thread to generate speech
    speech_thread = threading.Thread(target=generate_speech, args=(tts_text,))
    speech_thread.start()

    _, img_encoded = cv2.imencode('.jpg', img)  # Encode image to JPEG format
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')  # Convert to base64 string

    return objects_detected, img_base64

def generate_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file to the temporary directory
    _, temp_file_path = tempfile.mkstemp(suffix='.jpg', dir=temp_dir)
    file.save(temp_file_path)

    objects_detected, img_base64 = detect_objects(temp_file_path)

    # Response data
    response_data = {'objects_detected': objects_detected, 'image': img_base64}

    return jsonify(response_data)


@app.route('/summarize', methods=['POST'])
def summarize():
    # Get question text from form data
    question_text = request.form['questionText']

    # Get question file
    question_file = request.files['questionFile']
    question_file_content = question_file.read().decode('utf-8')
    print(question_file_content)
    # Get detected objects from the payload
    detected_objects = request.form['detectedObjects']

    # Generate content using GenerativeAI
    prompt=prompt = """
You are an AI assistant capable of answering user queries and incorporating detected objects from provided images for context. Users will provide a list of objects extracted from images, which you will integrate into your response. Users may also offer additional context to refine your answer.

Example Question:  
"How can I make a classic Italian spaghetti carbonara?"

Detected Objects (from image): eggs, bacon, pasta, cheese

Additional Context (optional):  
The user prefers a recipe that's easy to follow and doesn't require too many ingredients.

Response:  
"To prepare a classic Italian spaghetti carbonara, you'll need eggs, bacon, pasta, and cheese. Begin by boiling the pasta until it reaches al dente perfection. While the pasta cooks, whisk together eggs and cheese in a bowl, creating a creamy mixture. In a separate pan, crisp up the bacon until it's golden and crunchy. Once the pasta is cooked, drain it and immediately combine it with the egg and cheese mixture, allowing the residual heat to cook the eggs and form a rich, velvety sauce. Stir in the crispy bacon pieces for that perfect savory flavor. Remember, it's essential not to overcook the eggs; you want them to coat the pasta, not scramble. Serve your carbonara piping hot, topped with an extra sprinkle of cheese and a generous twist of black pepper. This recipe is wonderfully simple yet incredibly delicious, ideal for a satisfying Italian meal at home."
"""

    response = model.generate_content( prompt + ' now the actual user quer starts ---> Question : '+ question_text + '\nobjects from image : ' + detected_objects)

    # Convert response to string
    summary_text = str(response)
    print(response.candidates[0].content.parts[0].text)

    # Return the response
    return jsonify({'summary': response.candidates[0].content.parts[0].text, 'detected_objects': detected_objects})

if __name__ == '__main__':
    app.run(debug=True)

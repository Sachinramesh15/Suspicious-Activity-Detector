from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# Flask app
app = Flask(__name__)

# Load models
lstm_model = load_model('suspicious_activity_model.h5')
yolo_model = YOLO('yolov8n-pose.pt')  # Ensure this model supports keypoint detection
scaler = StandardScaler()

# Function to extract keypoints from a frame
def extract_keypoints(frame):
    """
    Extracts normalized keypoints from a frame using YOLO pose model.
    """
    results = yolo_model(frame, verbose=False)
    for r in results:
        if r.keypoints is not None and len(r.keypoints) > 0:
            # Extract the first detected person's keypoints
            keypoints = r.keypoints.xyn.tolist()[0]  # Use the first person's keypoints
            flattened_keypoints = [kp for keypoint in keypoints for kp in keypoint[:2]]  # Flatten x, y values
            return flattened_keypoints
    return None  # Return None if no keypoints are detected

# Function to process each frame
def process_frame(frame):
    # Perform YOLO detection
    results = yolo_model(frame, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])  # Class ID
        confidence = float(box.conf[0])

        # Detect persons only (class_id 0 for 'person')
        if cls == 0 and confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Extract ROI for classification
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Preprocess ROI to extract keypoints
                keypoints = extract_keypoints(roi)
                if keypoints is not None and len(keypoints) > 0:
                    # Standardize and reshape keypoints for LSTM input
                    keypoints_scaled = scaler.fit_transform([keypoints])  # Standardize features
                    keypoints_reshaped = keypoints_scaled.reshape((1, 1, len(keypoints)))  # Reshape for LSTM

                    # Predict with LSTM model
                    prediction = (lstm_model.predict(keypoints_reshaped) > 0.5).astype(int)[0][0]

                    # Draw bounding box and label
                    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                    label = 'Suspicious' if prediction == 1 else 'Normal'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    print("No valid keypoints detected for ROI. Skipping frame.")
            else:
                print("ROI size is zero. Skipping frame.")
    return frame


# Generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        frame = process_frame(frame)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main entry
if __name__ == "__main__":
    app.run(debug=True)

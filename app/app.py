from flask import Flask, render_template, Response, redirect, request, send_from_directory, url_for
import cv2
import os, time
import threading
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from ultralytics import YOLO

app = Flask(__name__)

# Directories
VIDEO_DIR = "C:/Users/sachi/Desktop/Capstone/suspicous-activity-detector/reinforcement_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Globals
lock = threading.Lock()
temporal_keypoints = []
TEMPORAL_WINDOW = 32
fps = 30

# Load models
yolo_model = YOLO("models/yolov8n-pose.pt")  # Load your YOLO model
lstm_model = load_model("models/lstm_nn.keras")  # Load your LSTM model

# Open camera
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Home Page."""
    return render_template('index.html')

video_files_list = []

# Modify the generate_camera_feed function to periodically save a video clip
def generate_camera_feed():
    """Capture, process, and stream camera feed."""
    global temporal_keypoints, video_files_list

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4
    current_video_path = os.path.join(VIDEO_DIR, f"video_{int(time.time())}.mp4")  # Save as .mp4
    out = cv2.VideoWriter(current_video_path, fourcc, fps, (640, 480))

    start_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Save the frame to the current video file every frame
        out.write(frame)

        # Process frame with YOLO
        results = yolo_model(frame, verbose=False)
        for r in results:
            if hasattr(r, "keypoints") and hasattr(r.keypoints, "xyn"):
                keypoints = r.keypoints.xyn.tolist()
                flattened_keypoints = [val for kp in keypoints[0] for val in kp]
                temporal_keypoints.append(flattened_keypoints)

                # Extract bounding boxes
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.tolist())

                    # Predict with LSTM when sufficient keypoints are available
                    if len(temporal_keypoints) >= TEMPORAL_WINDOW:
                        input_data = np.array(temporal_keypoints[-TEMPORAL_WINDOW:]).flatten()
                        input_data = input_data.reshape(1, TEMPORAL_WINDOW, -1)
                        prediction = lstm_model.predict(input_data, verbose=False)
                        predicted_class = "normal" if np.argmax(prediction) == 1 else "suspicious"

                        # Set color based on prediction
                        color = (0, 255, 0) if predicted_class == "normal" else (0, 0, 255)

                        # Draw bounding box and label
                        label = f"{predicted_class} ({prediction[0][np.argmax(prediction)]:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check if 2 minutes have passed (120 seconds)
        if time.time() - start_time >= 120:
            start_time = time.time()  # Reset the timer
            current_video_path = os.path.join(VIDEO_DIR, f"video_{int(time.time())}.avi")
            out.release()  # Release the old video writer
            out = cv2.VideoWriter(current_video_path, fourcc, fps, (640, 480))  # Start a new one

            # Add the new video to the list of videos
            video_files_list.append(current_video_path)

            # Keep only the most recent 5 videos
            if len(video_files_list) > 5:
                # Remove the oldest video if we have more than 5
                oldest_video = video_files_list.pop(0)
                os.remove(oldest_video)  # Delete the oldest video file

        # Encode frame as JPEG and yield it for the web feed
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    out.release()  # Ensure to release the video writer when the loop ends


@app.route('/video_feed')
def video_feed():
    """Video feed route."""
    return Response(generate_camera_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/reinforcement')
def reinforcement_page():
    """Reinforcement Page."""
    global video_files_list

    # Fetch the list of video files from the directory
    videos = [
        video for video in os.listdir(VIDEO_DIR)
        if video.endswith((".mp4", ".avi"))
    ]

    # Sort videos by modification time to show the most recent first
    videos = sorted(
        videos,
        key=lambda v: os.path.getmtime(os.path.join(VIDEO_DIR, v)),
        reverse=True
    )[:5]  # Limit to the 5 most recent videos

    message = request.args.get('message', '')  # Get the success message from URL params
    return render_template('reinforcement.html', videos=videos, message=message)

@app.route('/reinforce/<video_name>', methods=['POST'])
def reinforce(video_name):
    """Run reinforcement learning on the selected video."""
    video_path = os.path.join(VIDEO_DIR, video_name)
    run_reinforcement(video_path)
    
    # Return a confirmation message after reinforcement
    return redirect(url_for('reinforcement_page', message=f"Model Reinforced with {video_name}"))

@app.route('/serve_video/<filename>')
def serve_video(filename):
    """Serve video files."""
    return send_from_directory(VIDEO_DIR, filename)

def run_reinforcement(video_path):
    """Reinforce model using the opposite of the predicted class as feedback."""
    global lstm_model
    cap = cv2.VideoCapture(video_path)
    temporal_keypoints = []

    # Accumulate keypoints for the video
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with YOLO
        results = yolo_model(frame, verbose=False)
        for r in results:
            if hasattr(r, "keypoints") and hasattr(r.keypoints, "xyn"):
                keypoints = r.keypoints.xyn.tolist()
                flattened_keypoints = [val for kp in keypoints[0] for val in kp]
                temporal_keypoints.append(flattened_keypoints)

                if len(temporal_keypoints) >= TEMPORAL_WINDOW:
                    input_data = np.array(temporal_keypoints[-TEMPORAL_WINDOW:]).flatten()
                    input_data = input_data.reshape(1, TEMPORAL_WINDOW, -1)

                    # Predict class for the sequence
                    prediction = lstm_model.predict(input_data, verbose=False)
                    predictions.append(np.argmax(prediction))  # Store predicted classes

    cap.release()

    # Determine the majority predicted class for the video
    if predictions:
        predicted_class = max(set(predictions), key=predictions.count)
        true_label = 1 - predicted_class  # Inverse of the predicted class

        # Reinforce the model
        for i in range(len(temporal_keypoints) - TEMPORAL_WINDOW + 1):
            input_data = np.array(temporal_keypoints[i:i + TEMPORAL_WINDOW]).flatten()
            input_data = input_data.reshape(1, TEMPORAL_WINDOW, -1)

            with tf.GradientTape() as tape:
                predictions = lstm_model(input_data, training=True)
                reward = 1 if np.argmax(predictions) == true_label else -1
                loss = -reward

            gradients = tape.gradient(loss, lstm_model.trainable_variables)
            lstm_model.optimizer.apply_gradients(zip(gradients, lstm_model.trainable_variables))

    cap.release()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

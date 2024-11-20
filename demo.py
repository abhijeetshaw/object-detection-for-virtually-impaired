import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math
import pyttsx3
import threading
import queue

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load the TensorFlow model
model = hub.load("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow1/openimages-v4-ssd-mobilenet-v2/1").signatures["default"]

# Define color codes for detected classes
colorcodes = {}

# Dictionary to track object announcements and their counts
announcement_tracker = {}

# Function to draw bounding boxes and calculate distances
def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=2)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left, top - int(height * 6e-2)),
        (right, top),
        color=color,
        thickness=-1
    )
    cv2.putText(
        image,
        namewithscore,
        (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(255, 255, 255)
    )

# Function to calculate distance (example calculation based on bounding box size)
def calculate_distance(ymin, ymax):
    box_height = ymax - ymin
    distance = round(1 / box_height, 2)
    return distance

# Function to draw all detections
def draw(image, boxes, classnames, scores, tts_queue):
    global announcement_tracker
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=5, score_threshold=0.3)
    
    # Ensure boxesidx is not empty before proceeding
    if boxesidx.shape[0] == 0:  # Check if no boxes are detected
        announcement_tracker = {}  # Reset tracker if no objects are detected
        return image
    
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        distance = calculate_distance(ymin, ymax)
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)

        # Add to TTS queue only if not announced 2 times already
        if announcement_tracker.get(classname, 0) < 2:
            tts_queue.put(f"{classname} detected at approximately {distance:.2f} meters.")
            announcement_tracker[classname] = announcement_tracker.get(classname, 0) + 1

    return image

# Thread for Text-to-Speech
def tts_thread(tts_queue):
    while True:
        if not tts_queue.empty():
            text = tts_queue.get()
            engine.say(text)
            engine.runAndWait()

# Thread for video capture
def video_capture_thread(video_queue):
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if not video_queue.full():
            video_queue.put(frame)
    video.release()

# Thread for processing frames
def processing_thread(video_queue, tts_queue):
    while True:
        if not video_queue.empty():
            img = video_queue.get()
            img = cv2.resize(img, (640, 480))  # Resize for faster processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]
            detection = model(img_tensor)
            result = {key: value.numpy() for key, value in detection.items()}
            image_with_boxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"], tts_queue)
            cv2.imshow("Detection", image_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    video_queue = queue.Queue(maxsize=1)  # Limit the queue size to prevent lag
    tts_queue = queue.Queue()

    # Start threads
    threading.Thread(target=video_capture_thread, args=(video_queue,), daemon=True).start()
    threading.Thread(target=tts_thread, args=(tts_queue,), daemon=True).start()
    processing_thread(video_queue, tts_queue)

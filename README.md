# object-detection-for-virtually-impaired
## Overview
This project uses TensorFlow for real-time object detection and pyttsx3 for text-to-speech (TTS) to help visually impaired individuals navigate their environment by detecting objects and providing spoken feedback about their distance.

## Features
- Real-time object detection using a MobileNetV2 model.
- Text-to-speech feedback for detected objects.
- Distance estimation based on the size of the detected bounding box.
- Dynamic object announcement to avoid repetitive speech.

## Installation
1. **TensorFlow** - for object detection using pre-trained models.
   ```
   pip install tensorflow
   ```

2. **OpenCV** - for video capture and image processing.
   ```
   pip install opencv-python
   ```

3. **NumPy** - for numerical operations, especially in image processing.
   ```
   pip install numpy
   ```

4. **TensorFlow Hub** - to load the pre-trained object detection model.
   ```
   pip install tensorflow-hub
   ```

5. **pyttsx3** - for Text-to-Speech functionality.
   ```
   pip install pyttsx3
   ```

6. **threading** - This is part of Python's standard library, so no need to install it separately.

7. **queue** - This is also part of Python's standard library, so no separate installation is needed.

### Prerequisites
Ensure that you have Python 3.7 or higher installed.

### Required Libraries
Install the necessary libraries by running the following command:
```
pip install tensorflow opencv-python numpy tensorflow-hub pyttsx3
```

### Running the Program
1. Clone this repository to your local machine.
2. Install the required dependencies (as shown above).
3. Run the `object_detection.py` script using:
   ```
   python object_detection.py
   ```

4. The program will capture video from your webcam, process it, and announce detected objects with their approximate distance.

## Notes
- The script uses a TensorFlow Hub model (`mobilenet-v2` pre-trained on Open Images) for object detection.
- TTS feedback is provided only for new detections or when the object is detected with a sufficient distance.
- Press 'q' to quit the program at any time.

## License
This project is licensed under the MIT License.
```

### Steps to Run:
1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python numpy tensorflow-hub pyttsx3
   ```

3. **Run the script**:
   ```bash
   python object_detection.py
   ```

This `README.md` provides a clear overview of the project and installation steps, so users can easily set up and run the application.

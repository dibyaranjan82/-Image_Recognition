# -Image_Recognition
A full-stack AI-powered Image Recognition System built with Python, Flask, TensorFlow, PyTorch &amp; OpenCV — featuring Classification, Object Detection, and Batch Processing with a dark-themed web UI.
============================================================
  Image_Recognition — HOW TO RUN (Step by Step)
============================================================

STEP 1: Open Command Prompt (CMD)
   - Press  Win + R
   - Type   cmd
   - Press  Enter

STEP 2: Go to the project folder
   cd D:\ Image_Recognition
   (change D:\ Image_Recognition to wherever you put this folder)

STEP 3: Install packages (FIRST TIME ONLY)
   pip install flask flask-cors Pillow numpy opencv-python

STEP 4: Start the server
   python api_server.py

   You will see:
   ==================================================
      Image_Recognition is RUNNING!
     Open browser: http://localhost:5000
   ==================================================

STEP 5: Open your browser and go to:
   http://localhost:5000

That's it! The website will open.

------------------------------------------------------------
NEXT TIME (steps 1, 2, 4 only):
   cd D:\ Image_Recognition
   python api_server.py
   Then open http://localhost:5000
------------------------------------------------------------

PAGES IN THE APP:
   /            → Image Classification (upload photo → get labels)
   /detect      → Object Detection (find objects with boxes)
   /batch       → Batch (classify many images at once)
   /docs        → Documentation

NOTE: The app works WITHOUT TensorFlow/PyTorch installed.
It uses demo/mock data by default.
For real AI inference install:
   pip install tensorflow        (for classification)
   pip install torch torchvision (for YOLO detection)

============================================================

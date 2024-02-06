import os
import cv2
import numpy as np
import torch  # If using PyTorch for deep learning
from flask import Flask, request, jsonify  # For the Flask API

# Initialize Flask app
app = Flask(__name__)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

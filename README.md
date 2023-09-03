# Dynamic Hand Gesture Recognition
Dynamic Hand Gesture Recognition uses deep learning to recognize and classify different hand gestures from video data. The model is built using PyTorch and is based on the pretrained ResNet-3D architecture. This repository contains the structure of the dataset, code for preprocessing video data, training the model, and visualizing the results.

# Dataset
The dataset used for this project consists of 17 classes, with each class containing multiple video gestures.
The videos in this dataset capture hand landmarks that have been extracted using the MediaPipe framework. The code responsible for this extraction can be found in this repository under the filename Landmarks_extraction.py. One standout feature of this dataset is its uniformity in frame length across all videos. This ensures consistency during model training and evaluation. The script that accomplishes this frame resampling is named frame_resample.py and is also available in this repository.

# Prerequisites
Install the required packages using pip:

```python
conda install torch torchvision opencv-python numpy scikit-learn matplotlib mediapipe
```
# Run the Model
In the model code, ensure to change the dataset's root directory to its respective location on your system.

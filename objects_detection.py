# -*- coding: utf-8 -*-
"""yolo_algorithm

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O7CSzt6Ha8nlcIr2a-kkZ2wnoan2BZN6
"""

!git clone https://github.com/WongKinYiu/yolov9

!wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

!wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt

!wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

!wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt

# Commented out IPython magic to ensure Python compatibility.
# %cd yolov9/

!pip install -r requirements.txt -q
# Replace the source with your image or video path 
!python detect.py --weights '/content/gelan-e.pt' --source '/content/test_yolo1.jpg' --device 0 --classes 0

from IPython.display import Image
# if your using video replace image with video
Image(filename='runs/detect/exp/test_yolo.jpg')
# Replace the source with your image or video path 
!python detect.py --weights '/content/gelan-e.pt' --source '/content/test_yolo1.jpg' --device 0
# if your using video replace image with video
Image(filename='runs/detect/exp2/test_yolo1.jpg')
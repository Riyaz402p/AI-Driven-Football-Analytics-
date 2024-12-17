# %cd yolov9/

!pip install -r requirements.txt -q
# Replace the source with your image or video path 
!python detect.py --weights '/content/gelan-e.pt' --source '/content/test_yolo1.jpg' --device 0 --classes 0

from IPython.display import Image
# if your using video replace image with video
Image(filename='runs/detect/exp/test_yolo.jpg')
# Replace the source with your image or video path 
!python detect.py --weights '/content/gelan-e.pt' --source '/content/test_yolo1.jpg' --device 0
Image(filename='runs/detect/exp2/test_yolo1.jpg')
# if your using video replace image with video
Image(filename='runs/detect/exp2/test_yolo1.jpg')

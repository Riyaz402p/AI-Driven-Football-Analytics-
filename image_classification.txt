#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = r'C:\Users\varma\OneDrive\Desktop\assigment.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original grayscale image
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')
plt.show()

# Sobel gradient in x-direction
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
plt.figure(figsize=(8, 6))
plt.imshow(sobelx, cmap='gray')
plt.title("Sobel Gradient in X-direction")
plt.axis('off')
plt.show()

# Sobel gradient in y-direction
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
plt.figure(figsize=(8, 6))
plt.imshow(sobely, cmap='gray')
plt.title("Sobel Gradient in Y-direction")
plt.axis('off')
plt.show()

# Combine the gradients (magnitude)
edges = cv2.sqrt(sobelx**2 + sobely**2)
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.title("Edge Magnitude (Sobel Magnitude)")
plt.axis('off')
plt.show()

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(figsize=(8, 6))
plt.imshow(binary_image, cmap='gray')
plt.title("Binary Image (Otsu Thresholding)")
plt.axis('off')
plt.show()

# Perform morphological opening and closing operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)  # Opening
plt.figure(figsize=(8, 6))
plt.imshow(cleaned_image, cmap='gray')
plt.title("After Morphological Opening")
plt.axis('off')
plt.show()

# Perform closing operation
cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)  # Closing
plt.figure(figsize=(8, 6))
plt.imshow(cleaned_image, cmap='gray')
plt.title("After Morphological Closing")
plt.axis('off')
plt.show()

# Find contours
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_image = cv2.drawContours(cleaned_image.copy(), contours, -1, (255, 255, 255), 1)

# Display the final boundary detection
plt.figure(figsize=(8, 6))
plt.imshow(boundary_image, cmap='gray')
plt.title("Final Boundary Detection")
plt.axis('off')
plt.show()


# In[ ]:





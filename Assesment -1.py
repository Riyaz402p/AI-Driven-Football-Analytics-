import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image_file_path = r'C:\Users\varma\OneDrive\Desktop\assigment.jpg'
gray_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

# Display the original grayscale image
plt.figure(figsize=(8, 6))
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')
plt.show()

# Sobel gradient in x-direction
sobel_x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
plt.figure(figsize=(8, 6))
plt.imshow(sobel_x_gradient, cmap='gray')
plt.title("Sobel Gradient in X-direction")
plt.axis('off')
plt.show()

# Sobel gradient in y-direction
sobel_y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
plt.figure(figsize=(8, 6))
plt.imshow(sobel_y_gradient, cmap='gray')
plt.title("Sobel Gradient in Y-direction")
plt.axis('off')
plt.show()

# Combine the gradients (magnitude)
edge_magnitude = cv2.sqrt(sobel_x_gradient**2 + sobel_y_gradient**2)
plt.figure(figsize=(8, 6))
plt.imshow(edge_magnitude, cmap='gray')
plt.title("Edge Magnitude (Sobel Magnitude)")
plt.axis('off')
plt.show()

# Apply Otsu's thresholding
_, otsu_binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(figsize=(8, 6))
plt.imshow(otsu_binary_image, cmap='gray')
plt.title("Binary Image (Otsu Thresholding)")

plt.axis('off')
plt.show()

# Perform morphological opening and closing operations
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_opened_image = cv2.morphologyEx(otsu_binary_image, cv2.MORPH_OPEN, morph_kernel)  # Opening
plt.figure(figsize=(8, 6))
plt.imshow(morph_opened_image, cmap='gray')
plt.title("After Morphological Opening")
plt.axis('off')
plt.show()

# Perform closing operation
morph_closed_image = cv2.morphologyEx(morph_opened_image, cv2.MORPH_CLOSE, morph_kernel)  # Closing
plt.figure(figsize=(8, 6))
plt.imshow(morph_closed_image, cmap='gray')
plt.title("After Morphological Closing")
plt.axis('off')
plt.show()

# Find contours
contours, _ = cv2.findContours(morph_closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_image = cv2.drawContours(morph_closed_image.copy(), contours, -1, (255, 255, 255), 1)

# Display the final boundary detection
plt.figure(figsize=(8, 6))
plt.imshow(boundary_image, cmap='gray')
plt.title("Final Boundary Detection")
plt.axis('off')
plt.show()

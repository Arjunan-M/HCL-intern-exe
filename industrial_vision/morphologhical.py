import cv2
import matplotlib.pyplot as plt
import numpy as np
image_path = r"C:\Users\arjun\OneDrive\Pictures\morphology.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(
    image,cv2.MORPH_OPEN,kernel
)
closing = cv2.morphologyEx(
    opening,cv2.MORPH_CLOSE,kernel
)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Noisy Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(opening, cmap="gray")
plt.title("After Opening")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(closing, cmap="gray")
plt.title("After Closing")
plt.axis("off")
plt.tight_layout()
plt.show()
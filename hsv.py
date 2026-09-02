import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = r"C:\Users\arjun\OneDrive\Pictures\rgb.png"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([35, 40, 40])
upper_bound = np.array([85, 255, 255])

mask_hsv = cv2.inRange(hsv_image, lower_bound, upper_bound)
res_hsv = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_hsv)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_hsv, cmap='gray')
plt.title('HSV Binary Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(res_hsv)
plt.title('Segmented Output (HSV)')
plt.axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\arjun\OneDrive\Pictures\rgb.png"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

lower_bound = np.array([20, 0, 100])
upper_bound = np.array([255, 110, 200])

mask_lab = cv2.inRange(lab_image, lower_bound, upper_bound)
res_lab = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_lab)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_lab, cmap='gray')
plt.title('LAB Binary Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(res_lab)
plt.title('Segmented Output (LAB)')
plt.axis('off')

plt.tight_layout()
plt.show()

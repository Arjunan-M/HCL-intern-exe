import cv2
import matplotlib.pyplot as plt
image_path = r"C:\Users\arjun\OneDrive\Pictures\crack.jpg"
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

adaptive_mean = cv2.adaptiveThreshold(
    image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2
)
adaptive_gaussian = cv2.adaptiveThreshold(
    image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2
)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Uneven Illumination")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(adaptive_mean, cmap="gray")
plt.title("Adaptive Mean")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(adaptive_gaussian, cmap="gray")
plt.title("Adaptive Gaussian")
plt.axis("off")
plt.tight_layout()
plt.show()
import cv2
import matplotlib.pyplot as plt
image_path = r"C:\Users\arjun\OneDrive\Pictures\crack.jpg"
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)
threshold_value, binary = cv2.threshold(
    image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
print("Otsu Threshold Value:", threshold_value)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Otsu Defect Segmentation")
plt.axis("off")
plt.tight_layout()
plt.show()

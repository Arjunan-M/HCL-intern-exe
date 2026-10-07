import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu",input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_accuracy)
image_index = 0
prediction = model.predict(
    x_test[image_index:image_index + 1]
)
predicted_class = prediction.argmax()
print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[y_test[image_index][0]])
plt.imshow(x_test[image_index])
plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Actual: {class_names[y_test[image_index][0]]}"
)
plt.axis("off")
plt.show()

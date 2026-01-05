import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# -----------------------------
# Load and normalize MNIST data
# -----------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# -----------------------------
# Build the model
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=8)

# -----------------------------
# Interactive testing loop
# -----------------------------
def on_key(event):
    if event.key == 'c':
        print("Closing loop...")
        plt.close('all')
        sys.exit()  # stop the infinite loop

index = 0
num_samples = len(x_test)

while True:
    img_array = np.expand_dims(x_test[index], axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Display image
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)

    ax.imshow(x_test[index], cmap='gray')
    ax.set_title(f"Predicted: {predicted_label}, Actual: {y_test[index]}")
    ax.axis('off')

    print(f"Image {index}: Predicted: {predicted_label}, Actual: {y_test[index]}")
    plt.show()

    index = (index + 1) % num_samples  # loop infinitely over test images

from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "CIFAR10-CNN"

# Load CIFAR-10 data
(X, y), (X_test, y_test) = cifar10.load_data()

# For binary classification, let's take just "cats" (3) vs "dogs" (5)
import numpy as np
binary_classes = [3, 5]  # cat=3, dog=5
mask = np.isin(y, binary_classes).flatten()
X = X[mask]
y = y[mask]
y = (y == 5).astype(int)  # dog=1, cat=0

# Normalize
X = X/255.0

# Model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=32,
          epochs=3,
          validation_split=0.3,
          callbacks=[tensorboard])

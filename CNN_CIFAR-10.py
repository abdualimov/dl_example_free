"""
CNN для CIFAR-10

Датасет CIFAR-10 состоит из цветных картинок 32x32, разделенных на 10 классов:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("We're using TF", tf.__version__)
import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import models as M
print("We are using Keras", keras.__version__)
import pandas as pd

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# посмотрим на примеры картинок
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()

x_train2 = x_train.astype(np.float) / 255 - 0.5
x_test2 = x_test.astype(np.float) / 255 - 0.5
y_train2 = keras.utils.to_categorical(y_train, 10)
y_test2 = keras.utils.to_categorical(y_test, 10)
# конвертируем метки в np.array (?, NUM_CLASSES)

def make_model():
    model = M.Sequential()
    model.add(L.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3), activation='sigmoid'))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    model.add(L.BatchNormalization())
    model.add(L.MaxPool2D())
    model.add(L.Dropout(rate=0.25))
    model.add(L.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    model.add(L.BatchNormalization())
    model.add(L.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    model.add(L.BatchNormalization())
    model.add(L.MaxPool2D())
    model.add(L.Dropout(rate=0.25))
    model.add(L.Flatten())
    model.add(L.Dense(256, activation='sigmoid'))
    model.add(L.BatchNormalization())
    model.add(L.Dropout(rate=0.5))
    model.add(L.Dense(NUM_CLASSES, activation='softmax'))
    return model

K.clear_session()
model = make_model()
model.summary()

BATCH_SIZE = 32
EPOCHS = 10

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train2, y_train2,  # нормализованные данные
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test2, y_test2),
    shuffle=True
)

# тестовые предсказания
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

# confusion matrix и accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))


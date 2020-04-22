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
import numpy as np

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

kg = np.array([1, 34, 56, 67, 78], dtype = float)
funt = np.array([2.20462, 74.9572, 123.459, 147.71, 171.961], dtype = float)

test_kg = np.array([45, 547, 2, 7, 78890], dtype = float)
funt_kg = np.array([99.208, 1205.93, 4.40925, 15.4324, 173922.68], dtype = float)

def make_model():
    model = M.Sequential()
    model.add(L.Dense(units=4, input_shape=[1]))
    model.add(L.Dense(units=4))
    model.add(L.Dense(units=1))
    return model

K.clear_session()
model = make_model()
model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer='adam')

history = model.fit(
    kg, funt, epochs=500, verbose=False
)

pred = model.predict(test_kg)

plt.plot(history.history['loss'])

print(pred.reshape(1,5))
print(funt_kg)

mean_squared_error(funt_kg, pred)
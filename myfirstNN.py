import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#dataset
x_train = np.array([[200.0,17.0],
                    [120.0, 5.0],
                    [425.0,20.0],
                    [212.0,18.0]])
y_train = np.array([1,0,0,1])
model = Sequential([ Dense(units = 3, activation = 'sigmoid', name ='layer1'),
                    Dense(units = 1, activation = 'sigmoid', name = 'layer2')])
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
)
model.fit(x_train, y_train, epochs = 5)

x_test = np.array([[300.0,21.0],
                   [400.0,25.0]])
predictions = model.predict(x_test)
print(predictions)

for x in predictions:
    if x >= 0.5:
        print(1)
    else:
        print(0)    
import tensorflow as tf
import numpy as np

#single inputs with 1 neuron
# x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
# model.summary()

# model.compile(optimizer='sgd', loss='mean_squared_error')

# model.fit(x,y, epochs=1000)


# print(model.predict([10]))

#2 inputs with 1 neuron
x = np.array([(1.0, 2.0), (3.0, 1.0), (2.0, 2.0), (2.0, 3.0), (4.0, 2.0), (2.0, 0.0)], dtype=float)
y = np.array([5.0, 5.0, 6.0, 8.0, 8.0, 2.0], dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[2])])
model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x,y, epochs=1000)


print(model.predict([(2,4)]))

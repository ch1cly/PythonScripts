import tensorflow as tf
import numpy as np
def forward(x):
  cost = []
  z = tf.Variable(tf.zeros_like(x), trainable=False)
  s = tf.shape(x)[0]
  for i in range(s):
    z[i].assign(x[i]**i)
    cost.append(x[i]**i)
  return cost

a = tf.Variable(np.ones([5])*3)

with tf.GradientTape() as tape:
  b = forward(a)
grad = tape.gradient(b, a)
print(grad)
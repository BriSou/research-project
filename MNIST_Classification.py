import mnist

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

ed.set_seed(42)

def build_toy_dataset(x_train, y_train, N, pixel):
  x = (1/255)*x_train[0:N]
  x = tf.cast(x,tf.float32)
  x = tf.reshape(x,[N,pixel*pixel])
  y = y_train[0:N]
  y = tf.cast(y,tf.float32)
  return x, y

##Chargement des données train et test

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

Nb_train = train_images.shape[0]
Nb_test = test_images.shape[0]



##plt.imshow(x,cmap='gray')
##plt.show()

##Neural Network

def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return tf.reshape(h, [-1])

ed.set_seed(42)

N = 2000  # number of images train
pixel = 28
D = pixel*pixel   # number of features



x_train, y_train = build_toy_dataset(train_images,train_labels,N,pixel)

W_0 = Normal(loc=tf.zeros([D, 3]), scale=tf.ones([D, 3]))
W_1 = Normal(loc=tf.zeros([3, 1]), scale=tf.ones([3, 1]))
b_0 = Normal(loc=tf.zeros(3), scale=tf.ones(3))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = x_train

y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1),
           scale=0.1 * tf.ones(N))

qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [D, 3]),
              scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [D, 3])))
qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [3, 1]),
              scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [3, 1])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [3]),
              scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [3])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]),
              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y: y_train})
inference.run(n_iter=1000, n_samples=5)

# We use test data in order to test our neural network

x = (1/255)*train_images[1]
x = tf.cast(x,tf.float32)
x = tf.reshape(x,[1,pixel*pixel])
mus = tf.stack(
    [neural_network(x, qW_0.sample(), qW_1.sample(),
                    qb_0.sample(), qb_1.sample())
     for _ in range(100)])

sess = ed.get_session()
tf.global_variables_initializer().run()
outputs = mus.eval()
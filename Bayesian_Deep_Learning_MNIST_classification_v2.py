# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:46:58 2018

@author: soucheleaub
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd

#Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

N = 10   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.

I = 10     # number we delete from train data set to discover the outcome.

X_batch = np.array(np.zeros([N,D], dtype = np.float32))
Y_batch = np.array(np.zeros([N], dtype = np.int32))

# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(tf.matmul(x,w)+b)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter=100, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    i = 0
    while i < N:
        x_batch, y_batch = mnist.train.next_batch(1)
        # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
        y_batch = np.argmax(y_batch,axis=1)
        if (y_batch != I):
            X_batch[i] = x_batch
            Y_batch[i] = y_batch
            i += 1
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)



# Load the test images.
X_test = mnist.test.images[0:1000]
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels[0:1000],axis=1)

# Generate samples the posterior and store them.
n_samples = 1
prob_lst = []
samples = []
w_samples = []
b_samples = []
for n in range(n_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    # Also compue the probabiliy of each class for each   (w,b) sample.
    prob = tf.nn.softmax(tf.matmul( X_test,w_samp ) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)
    samples.append(sample.eval())
    print(n)
    
#Calibrage du modèle
prob_fin = prob_lst[0]    
#result récupère les digits prédit ainsi que la probabilité
#avec laquelle ils ont été prédits pour chaque image
result = np.zeros([1,1000,2], dtype=np.float32)
for i in range(0,1000):
    indice = np.argmax(prob_fin[i]).astype(np.int32)
    result[0][i][0] = indice
    result[0][i][1] = prob_fin[i][indice]

image_pred_0_6 = result[0][np.where((result[0][:,1] > 0.55) & (result[0][:,1] < 0.65))[0]]
image_pred_0_7 = result[0][np.where((result[0][:,1] > 0.65) & (result[0][:,1] < 0.75))[0]]
image_pred_0_8 = result[0][np.where((result[0][:,1] > 0.75) & (result[0][:,1] < 0.85))[0]]
image_pred_0_9 = result[0][np.where((result[0][:,1] > 0.85) & (result[0][:,1] < 0.95))[0]]
image_pred_1 = result[0][np.where((result[0][:,1] > 0.95) & (result[0][:,1] < 1.1))[0]]    

indice_pred_0_6 = np.where((result[0][:,1] > 0.55) & (result[0][:,1] < 0.65))[0]
indice_pred_0_7 = np.where((result[0][:,1] > 0.65) & (result[0][:,1] < 0.75))[0]
indice_pred_0_8 = np.where((result[0][:,1] > 0.75) & (result[0][:,1] < 0.85))[0]
indice_pred_0_9 = np.where((result[0][:,1] > 0.85) & (result[0][:,1] < 0.95))[0]
indice_pred_1 = np.where((result[0][:,1] > 0.95) & (result[0][:,1] < 1.1))[0]   


y_trn_prd_cali = np.argmax(prob_fin,axis=1).astype(np.float32)
calibrage_0_6 = (y_trn_prd_cali[indice_pred_0_6] == Y_test[indice_pred_0_6]).mean()*100 
calibrage_0_7 = (y_trn_prd_cali[indice_pred_0_7] == Y_test[indice_pred_0_7]).mean()*100 
calibrage_0_8 = (y_trn_prd_cali[indice_pred_0_8] == Y_test[indice_pred_0_8]).mean()*100 
calibrage_0_9 = (y_trn_prd_cali[indice_pred_0_9] == Y_test[indice_pred_0_9]).mean()*100     
calibrage_1 = (y_trn_prd_cali[indice_pred_1] == Y_test[indice_pred_1]).mean()*100     
# Compute the accuracy of the model.
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum proability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()


Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)


# Load the first image from the test data and its label.
indice = 0
#indice = np.where(Y_test == I)[0][0]
test_label = Y_test[indice]
test_image = X_test[indice]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.show()


# Now the check what the model perdicts for each (w,b) sample from the posterior. This may take a few seconds...
#sing_img_probs = []
#n_samples = 0
#for w_samp,b_samp in zip(w_samples,b_samples):
#    prob = tf.nn.softmax(tf.matmul(X_test[indice:(indice+1)],w_samp ) + b_samp)
#    sing_img_probs.append(prob.eval())
#    print(n_samples)
#    n_samples += 1

#
## Create a histogram of these predictions.
#plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(10))
#plt.xticks(np.arange(0,10))
#plt.xlim(0,10)
#plt.xlabel("Accuracy of the prediction of the test digit")
#plt.ylabel("Frequency")
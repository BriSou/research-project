
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, MultivariateNormalTriL, Normal
from edward.util import rbf
from observations import crabs

ed.set_seed(42)

data, metadata = crabs("~/data")
X_train = data[:100, 3:]
y_train = data[:100, 1]

N = X_train.shape[0]  # number of data points
D = X_train.shape[1]  # number of features

print("Number of data points: {}".format(N))
print("Number of features: {}".format(D))

X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalTriL(loc=tf.zeros(N), scale_tril=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)

qf = Normal(loc=tf.get_variable("qf/loc", [N]),
            scale=tf.nn.softplus(tf.get_variable("qf/scale", [N])))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=5000)
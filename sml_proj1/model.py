import keras
from keras.initializers import Constant
from keras.layers.core import Dense
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.regularizers import l1,l2
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import gc
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import csv


# https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras/50527423#50527423
def as_keras_metric(method):
    import functools
    from keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


auc_roc = as_keras_metric(tf.metrics.auc)
df = pd.read_csv('train_set.csv', header=0)
label = df['exist']
y = label.values
print([column for column in df])
df.drop(['source', 'target', 'exist'], axis=1, inplace=True)
x = df.values
print('input shape x is {}, y is {}'.format(x.shape, y.shape))
scaler = StandardScaler()
scaler.fit(x)
# print('scaler fit the data, mean is {}, var is {}'.format(scaler.mean_,scaler.var_))
del label
del df
# split train and dev sets
X_train, X_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=78, stratify=y)
del x
del y
# read test set
test_csv = pd.read_csv('test_set.csv', header=0)
test_csv.drop(['source', 'target'], axis=1, inplace=True)
X_test = test_csv.values
del test_csv
gc.collect()
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)
print('Xtrain shape is {}. Xdev shape is {}. Xtest shape is {}'.format(X_train.shape, X_dev.shape, X_test.shape))
# -----------------------------------------hypermeters--------------------------------------
optimizer = 'Adam'
learning_rate = 0.01
decay = 1e-6
momentum = 0.8
layer_weight_initializer_min = -0.005
layer_weight_initializer_max = 0.005
bias_initializer = 0.001
beta_1 = 0.95
beta_2 = 0.999
epsilon = 1e-08
adam = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
activation_function = 'sigmoid'
layer_amount = 3
batch_size = 3200
iteration_times = 5000
# record parameters
localtime = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
txt_name = str(localtime + '.txt')
# hypermeters objects
random_initializer = RandomUniform(minval=layer_weight_initializer_min, maxval=layer_weight_initializer_max)
bias_init = Constant(value=bias_initializer)
# --------------------------------------hypermeters end-------------------------------------
# model initialization
model = Sequential()
model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), use_bias=True, kernel_initializer=random_initializer,
                activation='tanh',bias_initializer=bias_init,kernel_regularizer=l2(0.001)))
model.add(Dense(X_train.shape[1], use_bias=True, kernel_initializer=random_initializer, activation='tanh',
                bias_initializer=bias_init,kernel_regularizer=l2(0.001)))
model.add(Dense(1, use_bias=False, kernel_initializer=random_initializer))
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
print("start training")
model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=iteration_times, verbose=2,
          validation_data=(X_dev, y_dev),
          callbacks=[TensorBoard(log_dir='log/assone')])
y_pred = model.predict(X_dev, batch_size=1, verbose=0)
with open(txt_name, 'w', newline="\n") as f:
    f.write('hypermeters:')
    f.write('optimizer: {}'.format(optimizer))
    f.write('learning_rate: {}'.format(learning_rate))
    f.write('decay: {}'.format(decay))
    f.write('momentum: {}'.format(momentum))
    f.write('weight initialized with min={} max={}'.format(layer_weight_initializer_min, layer_weight_initializer_max))
    f.write('bias initialized as {}'.format(bias_initializer))
    f.write('activation function is {}'.format(activation_function))
    # f.write('regularation method is {}, lambda is {}'.format(weight_regularizer_type,weight_regularizer))
    f.write('-------------------------------')
    f.write('model parameters:')
    f.write('layer numbers {}'.format(layer_amount))
    f.write('batch size {}'.format(batch_size))
    f.write('iteration times {}'.format(iteration_times))
    f.write('result: {}'.format(roc_auc_score(y_dev, y_dev)))
filename = localtime + '.h5'
model.save(filename)
print(model.summary())
y_answer = model.predict(X_test, batch_size=1, verbose=0)
with open('result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Prediction'])
    for i in range(X_test.shape[0]):
        writer.writerow([i + 1, y_answer[i]])

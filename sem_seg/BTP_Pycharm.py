import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

import h5py
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

ALL_FILES = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/all_files.txt')]#provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = load_h5(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

cat_labels = tf.keras.utils.to_categorical(label_batches, num_classes = 13, dtype = "uint8")
X_train, X_test, Y_train, Y_test = train_test_split( data_batches, cat_labels , test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

input_image = keras.Input(shape = (4096,9,1))
num_point = 4096
x = Conv2D(64, [1,9], padding='VALID', strides = [1,1], name ='conv1')(input_image)
print(x.shape)
x = Conv2D(64, [1,1], padding='VALID', strides = [1,1], name ='conv2')(x)
x = Conv2D(64, [1,1], padding='VALID', strides = [1,1], name ='conv3')(x)
x = Conv2D(128, [1,1], padding='VALID', strides = [1,1], name ='conv4')(x)
x= Conv2D(1024, [1,1], padding='VALID', strides = [1,1], name ='conv5')(x)
    # MAX
print(x.shape)
pc_feat1 = MaxPooling2D([num_point,1], padding='VALID', name ='maxpool1')(x)
print(pc_feat1.shape)
pc_feat1 = Flatten()(pc_feat1)
print(pc_feat1.shape)
#model.add(Flatten())
pc_feat1 = Dense(256, name='fc1')(pc_feat1)
pc_feat1 = Dense(128 , name='fc2')(pc_feat1)
print(pc_feat1.shape)
# CONCAT
a = tf.keras.layers.Reshape((1, 1, -1))(pc_feat1)
print(a.shape)

pc_feat1_expand = tf.tile(a, [1, num_point, 1, 1])
print(pc_feat1_expand.shape)
points_feat1_concat = tf.concat(axis=3, values=[x, pc_feat1_expand])
print(points_feat1_concat.shape)
# CONV

y = Conv2D(512, [1, 1], padding='VALID', strides=[1, 1], name='conv6')(points_feat1_concat)
y = Conv2D(256, [1, 1], padding='VALID', strides=[1, 1], name='conv7')(y)
# tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)

y = Dropout(rate=0.3)(y)
y = Conv2D(13, [1, 1], padding='VALID', strides=[1, 1], name='conv8')(y)
net = tf.squeeze(y, [2])

# return net
model = keras.Model(inputs=input_image, outputs=net)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)

model.fit(X_train, Y_train, batch_size=24, epochs=3, verbose=1)
model.evaluate(X_test, Y_test, verbose=2)

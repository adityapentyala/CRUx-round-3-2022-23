"""

"""
import copy

import tensorflow as tf
import keras_tuner
import numpy as np
import random
import matplotlib.pyplot as plot
import time

cifar100 = tf.keras.datasets.cifar100
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train, X_test = X_train / 255, X_test / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# taking 33% of X_train, i.e., 16500 samples
X_train_labelled = np.array(X_train[:16500])
y_train_labelled = np.array(y_train[:16500])
X_train_unlabelled = np.array(X_train[16500:])
y_train_psuedolabels = np.zeros(33500)

# print(y_train_labelled[:150])

# building the model
'''input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
mpool1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv1)
conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(mpool1)
conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(conv2)
mpool2 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')(conv3)
conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='relu')(mpool2)
mpool3 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')(conv4)
conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding='same', activation='relu')(mpool3)
mpool4 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv5)
flatten = tf.keras.layers.Flatten()(mpool4)
FC1 = tf.keras.layers.Dense(4096, activation='relu')(flatten)
FC2 = tf.keras.layers.Dense(1024, activation='relu')(FC1)
output_layer = tf.keras.layers.Dense(100, activation='softmax')(FC2)'''


def build_model(hp):
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    mpool1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(mpool1)
    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(conv2)
    mpool2 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')(conv3)
    conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='relu')(mpool2)
    mpool3 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same')(conv4)
    conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding='same', activation='relu')(mpool3)
    mpool4 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv5)
    flatten = tf.keras.layers.Flatten()(mpool4)
    FC1 = tf.keras.layers.Dense(units=hp.Choice("units_1", [1024, 2048, 4096]), activation='relu')(flatten)
    FC2 = tf.keras.layers.Dense(units=hp.Choice("units_2", [512, 1024, 2048]), activation='relu')(FC1)
    output_layer = tf.keras.layers.Dense(100, activation='softmax')(FC2)
    lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="acc",
    max_trials=5,
    executions_per_trial=1,
    overwrite=True
)

tuner.search(X_train_labelled, y_train_labelled, epochs=3)
best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])
model.summary()

'''model = build_model(keras_tuner.HyperParameters)
model.summary()'''

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
threshold = 0.5


class StopTraining(tf.keras.callbacks.Callback):
    def __init__(self):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.prev = 0

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') is not None and (logs.get('val_acc') - self.prev < 0.001):
            self.model.stop_training = True
            print("\n Model stopped training due to drop in validation accuracy")
        self.prev = logs.get('val_acc')


callbacks = StopTraining()


def psuedolabel_new_data(x, threshold):
    new_x, new_y = [], []
    deleted_x = []
    indices = []
    preds = model.predict(x)
    print("Preds shape", preds.shape)
    for index in range(len(preds)):
        if preds[index][np.argmax(preds[index])] > threshold:
            new_x.append(x[index])
            binary_pred = np.zeros(100)
            binary_pred[np.argmax(preds[index])] = 1.
            new_y.append(binary_pred)
            indices.append(index)
        else:
            deleted_x.append(x[index])
    return new_x, new_y, deleted_x


iters = 7

train_accuracy = []
test_accuracy = []

hist = model.fit(X_train_labelled, y_train_labelled, epochs=5, batch_size=64, validation_data=(X_test, y_test),
                 callbacks=[callbacks])
train_accuracy.extend(hist.history['acc'])
test_accuracy.extend(hist.history['val_acc'])
new_x, new_y, new_X_train_unlabelled = psuedolabel_new_data(X_train_unlabelled, threshold)
new_x = np.array(new_x)
new_y = np.array(new_y)
new_X_train_unlabelled = np.array(new_X_train_unlabelled)
X_train_labelled = np.append(X_train_labelled, new_x, axis=0)
y_train_labelled = np.append(y_train_labelled, new_y, axis=0)
X_train_unlabelled = new_X_train_unlabelled
print("New datapoints added:", new_x.shape, new_y.shape)
print("New size of labelled sets:", X_train_labelled.shape, y_train_labelled.shape)
while len(new_x) > 0 and iters > 0:
    hist = model.fit(X_train_labelled, y_train_labelled, epochs=5, batch_size=64, validation_data=(X_test, y_test),
                     callbacks=[callbacks])
    train_accuracy.extend(hist.history['acc'])
    test_accuracy.extend(hist.history['val_acc'])
    new_x, new_y, new_X_train_unlabelled = psuedolabel_new_data(X_train_unlabelled, threshold)
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    new_X_train_unlabelled = np.array(new_X_train_unlabelled)
    X_train_labelled = np.append(X_train_labelled, new_x, axis=0)
    y_train_labelled = np.append(y_train_labelled, new_y, axis=0)
    X_train_unlabelled = new_X_train_unlabelled
    print("New datapoints added:", new_x.shape, new_y.shape)
    print("New size of labelled sets:", X_train_labelled.shape, y_train_labelled.shape)
    if threshold < 0.75:
        threshold += 0.1
    iters -= 1

print()
print("Data plotted:")
print(train_accuracy)
print(test_accuracy)
epochs = range(1, len(train_accuracy) + 1)
print(epochs)
plot.figure()
plot.plot(epochs, train_accuracy, label='Training accuracy')
plot.plot(epochs, test_accuracy, label='Testing accuracy')
plot.title('Accuracy over epochs')
plot.xlabel('Epochs')
plot.ylabel('Accuracy')
plot.legend()
plot.show()

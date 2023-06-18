from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
from keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from pathlib import Path
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import os, os.path
from keras.applications import ResNet50V2, ResNet101V2, ResNet152V2, InceptionResNetV2
import keras
from keras.layers import Layer


train = Path('D:/Projects/pycharm project2/input/train')
train_filepaths = list(train.glob('**/*.[jp][pn]g'))
valid = Path('D:/Projects/pycharm project2/input/validation')
valid_filepaths = list(valid.glob('**/*.[jp][pn]g'))
test = Path('D:/Projects/pycharm project2/input/test')
test_filepaths = list(test.glob('**/*.[jp][pn]g'))

def process_img(filepath):

    labels = [str(filepath[i]).replace("/", "\\").split("\\")[-2]
              for i in range(len(filepath))]
    filepath = pd.Series(filepath, name='FilePath').astype(str)
    labels = pd.Series(labels, name='Label')

    df = pd.concat([filepath, labels], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)

    return df
train_df = process_img(train_filepaths)
valid_df = process_img(valid_filepaths)
test_df = process_img(test_filepaths)
train_df.head()
print('-- Training set --\n')
print(f'Number of pictures: {train_df.shape[0]}\n')
print(f'Number of different labels: {len(set(train_df.Label))}\n')
print(f'Labels: {set(train_df.Label)}')
unique_labels = train_df.copy().drop_duplicates(subset=['Label']).reset_index()

nrows = len(unique_labels)
ncols = 1
while nrows > 6:
    ncols += 1
    nrows = (len(unique_labels) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in zip(unique_labels.index, axes.flat):
    ax.imshow(plt.imread(unique_labels.FilePath[i]))
    ax.set_title(unique_labels.Label[i], fontsize=12)

plt.tight_layout()
plt.show()
# Fetch and process the data
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.inception_resnet_v2.preprocess_input
)
train_images = data_gen.flow_from_dataframe(
    dataframe = train_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valid_images = data_gen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(240,240,3),
    pooling='avg',
)
base_model.trainable = False

inputs = base_model.input

x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)

outputs = tf.keras.layers.Dense(14, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_images,
    validation_data=valid_images,
    batch_size = 32,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)
model.save("recoo.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Acc','Val'], loc = 'lower right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss','Val'], loc = 'upper right')
filenames = train_images.filenames
nb_samples = len(filenames)

loss, acc = model.evaluate(train_images,steps = (nb_samples), verbose=1)
print('accuracy test: ',acc)
print('loss test: ',loss)


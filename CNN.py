import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from pathlib import Path
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import keras as k
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

# config = tf.ConfigProto(device_count = {'GPU': 1})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Path to folder with the data
train_path = "/home/sm16709/kaggle/unziped/ILSVRC/Data/CLS-LOC/train/"


# create a data generator
datagen = ImageDataGenerator(validation_split=0.2, channel_shift_range=10, shear_range=0.15,
					zoom_range=0.2, horizontal_flip=True, rotation_range=10,
					brightness_range=[0.2,1.0],width_shift_range=0.1, height_shift_range=0.1)

print(device_lib.list_local_devices())

batch_size = 128

# number of files in a directory
def get_total_files(dir_path):
    total_files = 0
    for root, dirs, files in os.walk(dir_path):
        total_files += len(files)
    return total_files


# load and iterate the data sets
train_batches = datagen.flow_from_directory(train_path, class_mode='categorical', target_size=(128, 128),
                                            batch_size=batch_size, shuffle=True, subset="training")
valid_batches = datagen.flow_from_directory(train_path, class_mode='categorical', target_size=(128, 128),
                                            batch_size=batch_size, shuffle=True, subset="validation")

# confirm the iterator works
# batchX, batchy = train_batches.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


# ///// Build and fine-tune vgg16 model //////

vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# vgg16_model.summary()

# transform the model to sequential
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# exclude layers from future training
#for layer in model.layers:
#    layer.trainable = False

# add a final dense layer with output of 1000 (for 1000 category)
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1000, activation="softmax"))

model.summary()

# /////// train the vgg16 model //////

parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.compile(
              SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

train_steps = train_batches.samples / batch_size
valid_steps = valid_batches.samples / batch_size

parallel_model.fit_generator(train_batches, steps_per_epoch=train_steps, validation_data=valid_batches,
                    validation_steps=valid_steps, epochs=40, verbose=1, workers=6, use_multiprocessing=False)


# save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# save neural network's trained weights
model.save_weights("model_weights.h5")

# save the whole model
model.save("whole_model.h5")

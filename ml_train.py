from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/Users/satwiknaik/Desktop/Luminosity submissions/fire-detection-master/FIRE-SMOKE-DATASET/Train"

training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      zoom_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

VALIDATION_DIR = "/Users/satwiknaik/Desktop/Luminosity submissions/fire-detection-master/FIRE-SMOKE-DATASET/Test"
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224, 224),
    shuffle=True,
    class_mode='categorical',
    batch_size=128
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=True,
    batch_size=14
)

input_tensor = Input(shape=(224, 224, 3))

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') <= 0.1099 and logs.get('loss') <= 0.1099):
            print('\n\n Reached The Destination!')
            self.model.stop_training = True


callbacks = myCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=14,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=14,
    callbacks=[callbacks]
)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') <= 0.1099 and logs.get('loss') <= 0.1099):
            print('\n\n Reached The Destination!')
            self.model.stop_training = True


callbacks = myCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=14,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=14,
    callbacks=[callbacks]
)
print(len(base_model.layers))
model.save('InceptionV3.h5')

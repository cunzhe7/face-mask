import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

spath = r"C:\Users\dell\Desktop\shared\670\FinalProject\Face_Mask_Dataset\Train"
mpath = r"C:\Users\dell\Desktop\shared\670\FinalProject\Face_Mask_Dataset\Train\WithMask"
npath = r"C:\Users\dell\Desktop\shared\670\FinalProject\Face_Mask_Dataset\Train\WithoutMask"

imgs, labels = [], []

batchSize = 32
epochNum = 20

data_gen = ImageDataGenerator(
                        rescale = 1./255,
                        shear_range=0.2,
                        validation_split=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

train_gen = data_gen.flow_from_directory(
                        spath,
                        target_size=(224,224),
                        batch_size=batchSize,
                        class_mode="categorical",
                        subset="training"
                        )

valid_gen = data_gen.flow_from_directory(
                        spath,
                        target_size=(224,224),
                        batch_size=batchSize,
                        class_mode="categorical",
                        subset="validation"
                        )


model = Sequential()
MobNet = MobileNetV2((224,224,3), include_top=False)
model.add(MobNet)
model.add(AveragePooling2D((7,7)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics ="accuracy")


his = model.fit_generator(generator=train_gen,
                              steps_per_epoch=len(train_gen)//batchSize,
                              validation_data=valid_gen,
                              validation_steps=len(valid_gen) // batchSize,
                              epochs=epochNum
                              )
model.save("mask_detector.h5", save_format='h5')

plt.style.use("ggplot")
plt.figure()
plt.plot(range(epochNum), his.history['loss'], label = "loss")
plt.plot(range(epochNum), his.history['accuracy'], label = "accuracy")

plt.show()


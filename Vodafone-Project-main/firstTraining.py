import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import models, layers

TF_CPP_MIN_LOG_LEVEL = 2


def prep_dataset(X, y):
    X_prep = X.astype('float32')
    y_prep = to_categorical(np.array(y))
    return (X_prep, y_prep)


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = 'data/' + y_test["Path"].values
    data = []
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    X_test = np.array(data)
    return X_test, label


imgs_path = "data/Train"
data_list = []
labels_list = []
classes_list = 43
for i in range(classes_list):
    i_path = os.path.join(imgs_path, str(i))  # 0-42
    for img in os.listdir(i_path):
        im = Image.open(i_path + '/' + img)
        im = im.resize((32, 32))
        im = np.array(im)
        data_list.append(im)
        labels_list.append(i)
data = np.array(data_list)
labels = np.array(labels_list)

X, y = prep_dataset(data, labels)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5, shuffle=True)

model = models.Sequential()  # Sequential Model

# ConvLayer(64 filters) + MaxPooling + BatchNormalization + Dropout
model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=X.shape[1:]))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

# ConvLayer(128 filters) + MaxPooling + BatchNormalization + Dropout
model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

# ConvLayer(512 filters) + Dropout + ConvLayer(512 filters) + MaxPooling + BatchNormalization
model.add(layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())

# Flatten
model.add(layers.Flatten())

# 2 Dense layers with 4000 hidden units
model.add(layers.Dense(4000, activation='relu'))
model.add(layers.Dense(4000, activation='relu'))

# Dense layer with 1000 hidden units
model.add(layers.Dense(1000, activation='relu'))

# Softmax layer for output
model.add(layers.Dense(43, activation='softmax'))

print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# history = model.fit(X_train, Y_train,
#                     epochs=20,
#                     batch_size=64,
#                     validation_data=(X_val, Y_val))
history = model.fit(X_train, Y_train,
                    epochs=30, batch_size=64,
                    validation_data=(X_val, Y_val))

model.save(f"signsClassificationModel2.h5")


# model.save(f"signsClassificationModel_{accuracy}.h5")
def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = 'data/' + y_test["Path"].values
    data = []
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    X_test = np.array(data)
    return X_test, label


X_test, label = testing(r'data/Test.csv')
Y_pred = model.predict(X_test)
print(Y_pred)

print(accuracy_score(label, Y_pred))

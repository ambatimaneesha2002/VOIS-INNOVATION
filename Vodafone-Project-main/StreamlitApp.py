import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
from utils import class_labels as labels
import cv2

model = load_model('ClassificationModel.h5')


# def test_on_img(image):
#     data = []
#     image = image.resize((32, 32))
#     data.append(np.array(image))
#     X_test = np.array(data)
#     predict_x = model.predict(X_test)
#     Y_pred = np.argmax(predict_x, axis=1)
#     return Y_pred

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def test_on_img(image):
    img = np.asarray(image)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = np.argmax(model.predict(img), axis=1)
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    return classIndex, str(probVal * 100)


def run():
    st.title("Sign Board 🚧🚧🚧🚧 Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    print(img_file)
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        # pillow image to cv2 image
        st.image(img, use_column_width=False)
        # result = test_on_img(img)
        idx, prob = test_on_img(img)
        # s = [str(i) for i in result]
        # a = int("".join(s))
        print("Predicted traffic sign is: ", labels[idx[0]])
        st.success("**Predicted : " + labels[idx[0]] + '**')
        st.success("**Confidence is : " + prob + '**')


run()

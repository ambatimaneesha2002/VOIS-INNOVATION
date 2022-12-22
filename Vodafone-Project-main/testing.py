# from keras.models import load_model
# from time import sleep
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np
# from classLables import class_labels
# classifier = load_model('Emotion_Model.h5')
#
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     labels = []
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     roi = img_to_array(gray)
#     roi = np.expand_dims(roi, axis=0)
#
#     preds = classifier.predict(roi)[0]
#     label = class_labels[preds.argmax()]
#     label_position = (100, 100)
#     cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#     cv2.imshow('Emotion Detector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#######################################################

# import pandas as pd
# from keras.models import load_model
# from utils import class_labels as classes
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#
# model = load_model('./training/TSR1.h5')
#
#
# def testing(testcsv):
#     y_test = pd.read_csv(testcsv)
#     label = y_test["ClassId"].values
#     imgs = 'data/' + y_test["Path"].values
#     data = []
#     for img in imgs:
#         image = Image.open(img)
#         image = image.resize((30, 30))
#         data.append(np.array(image))
#     X_test = np.array(data)
#     return X_test, label
#
#
# def test_on_img(img):
#     data = []
#     image = Image.open(img)  # .convert('LA')
#     image = image.resize((30, 30))
#     data.append(np.array(image))
#     X_test = np.array(data)
#     # Y_pred = model.predict_classes(X_test)
#     predict_x = model.predict(X_test)
#     Y_pred = np.argmax(predict_x, axis=1)
#     score = tf.nn.softmax(predict_x[0])
#     print(100 * np.max(score))
#     return image, Y_pred, score
#
#
# plot, prediction, score = test_on_img(r'uploads/test3.jpg')
# s = [str(i) for i in prediction]
#
# a = int("".join(s))
# print("Predicted traffic sign is: ", classes[a])
# plt.imshow(plot)
# plt.show()

# t, l = testing(r'data/Test.csv')
# print(len(l))
# pred = model.predict(t)
# Y_pred = np.argmax(pred, axis=1)
# print(Y_pred)
# print(accuracy_score(l, Y_pred))


#######################################################
# from utils import class_labels as labels
# import cv2
# import numpy as np
# from keras.models import load_model
#
# model = load_model("ClassificationModel.h5")
# # imgOriginal = cv2.imread('uploads/test1.jpg')
# # imgOriginal = cv2.resize(imgOriginal, (320, 320))
# # img = np.asarray(imgOriginal)
# # img = cv2.resize(img, (32, 32))
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # img = cv2.equalizeHist(img)
# # img = img / 255
# # cv2.imshow("window", img)
# # cv2.waitKey(0)
# # img = img.reshape(1, 32, 32, 1)
# # classIndex = np.argmax(model.predict(img), axis=1)
# # predictions = model.predict(img)
# # probVal = np.amax(predictions)
# # print(classIndex, probVal)
# # imgOriginal = cv2.putText(imgOriginal, labels[classIndex[0]],
# #                           (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
# # print(labels[classIndex[0]])
# # cv2.imshow("window", imgOriginal)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cap = cv2.VideoCapture(0)
# while True:
#     _, imgOriginal = cap.read()
#     imgOriginal = cv2.resize(imgOriginal, (320, 320))
#     img = np.asarray(imgOriginal)
#     img = cv2.resize(img, (32, 32))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img / 255
#     img = img.reshape(1, 32, 32, 1)
#     classIndex = np.argmax(model.predict(img), axis=1)
#     predictions = model.predict(img)
#     probVal = np.amax(predictions)
#     print(classIndex, probVal)
#     if int(probVal * 100) > 80:
#         imgOriginal = cv2.putText(imgOriginal, labels[classIndex[0]],
#                                   (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
#         print(labels[classIndex[0]])
#     cv2.imshow("window", imgOriginal)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#######################################################
from utils import class_labels as labels
import cv2
import numpy as np
from keras.models import load_model

model = load_model("roadsModel.h5")
x = cv2.imread('testroad1.jpg')
x = cv2.resize(x, (256, 256))

# cv2.imshow("img", x)
# cv2.waitKey(0)

X = np.array(x)
X = np.expand_dims(X, axis=0)

# y_pred = np.round(model.predict(X))

classIndex = np.argmax(model.predict(X), axis=1)
predictions = model.predict(X)
print(classIndex)
print(int(100 * predictions[classIndex[0]][0]))
# if y_pred[0][0] == 1:
#     print("Plain Road")
# else:
#     print("Pothole Road")

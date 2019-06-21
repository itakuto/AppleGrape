from tensorflow.keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


class_names=['apple','grape']
files_apple=os.listdir('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Apple')
files_grape=os.listdir('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Grape')

X=[]
Y=[]

for i in files_apple:
    img=Image.open('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Apple/'+i)
    img=img.resize((28,28))
    X.append(img)
    Y.append(0)

for i in files_grape:
    img=Image.open('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Grape/'+i)
    img=img.resize((28,28))
    X.append(img)
    Y.append(1)

fig, ax = plt.subplots(4, 5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(20):
    img = X[i].resize((28, 28))
    ax[i].imshow(img)
    ax[i].set_title(i + 1)

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

for i in range (20):
    X[i]=img_to_array(X[i])/255

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train=np.array(X_train)
X_test=np.array(X_test)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28,3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20)

model.save('AppleGrape_model.h5')

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(X_test)

num_rows = 2
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, Y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, Y_test)
plt.show()

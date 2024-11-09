import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def load_img(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_or_train(model_path='cifar10_classifier.h5'):
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training model...")
        model = train_model()
        model.save(model_path)
        print(f"Model saved to {model_path}")
    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    num_classes = len(set(y_train.flatten()))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    return model

def predict_class(model, img_array):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = labels[predicted_class]
    return predicted_label

def show_result(img_path, predicted_label):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.show()

def predict(img_path):
    img_array = load_img(img_path)
    model = load_or_train()
    predicted_label = predict_class(model, img_array)
    show_result(img_path, predicted_label)

def main():
    # This is a test, you can also try it with other images. 
    # These are all google search images, not from CIFAR-10.
    img_path = 'img_truck.jpg'
    predict(img_path)

main()

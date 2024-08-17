import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define constants
IMAGE_SIZE = (224, 224)
model_path = r'C:/Users/hp/Desktop/CV CP - Object Grip Detection/prosthetic_hand_grasp_model.h5'
image_path = r'C:\Users\hp\Desktop\CV CP - Object Grip Detection\bottle\B_0030.jpg'  # Provide the path to your single test image here

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load and preprocess the single test image
img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255

# Predict the class of the single test image
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
class_labels = ['bottle', 'cup', 'mobile']  # Assuming these are the class labels used during training

# Plot the single test image with the predicted class label
plt.imshow(img)
plt.title(f"Predicted class: {class_labels[predicted_class_idx]}")
plt.axis('off')
plt.show()

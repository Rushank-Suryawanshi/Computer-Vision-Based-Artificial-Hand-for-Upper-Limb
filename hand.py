
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Define data directory
data_dir = 'C:/Users/hp/Desktop/CV CP - Object Grip Detection'

# Define subdirectories for classes
bottle_dir = os.path.join(data_dir, 'bottle')
cup_dir = os.path.join(data_dir, 'cup')
mobile_dir = os.path.join(data_dir, 'mobile')

# Create lists to store file paths and labels
file_paths = []
labels = []

# Iterate over bottle images and append file paths and labels
for filename in os.listdir(bottle_dir):
    if filename.endswith(".jpg"):
        file_paths.append(os.path.join(bottle_dir, filename))
        labels.append('bottle')

# Iterate over cup images and append file paths and labels
for filename in os.listdir(cup_dir):
    if filename.endswith(".jpg"):
        file_paths.append(os.path.join(cup_dir, filename))
        labels.append('cup')

# Iterate over mobile images and append file paths and labels
for filename in os.listdir(mobile_dir):
    if filename.endswith(".jpg"):
        file_paths.append(os.path.join(mobile_dir, filename))
        labels.append('mobile')

# Split data into training and testing sets (70% training, 30% testing)
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=42)

# Further split training set into training and validation sets (50% training, 50% validation)
train_files, valid_files, train_labels, valid_labels = train_test_split(train_files, train_labels, test_size=0.5, random_state=42)

# Print the number of samples in each set
print("Number of training samples:", len(train_files))
print("Number of validation samples:", len(valid_files))
print("Number of testing samples:", len(test_files))

# Data preprocessing for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values to range [0,1]
    rotation_range=20,          # Rotate images randomly up to 20 degrees
    width_shift_range=0.2,      # Shift images horizontally up to 20% of the width
    height_shift_range=0.2,     # Shift images vertically up to 20% of the height
    shear_range=0.2,            # Shear intensity
    zoom_range=0.2,             # Zoom range
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Fill mode for filling in newly created pixels
)

# Data preprocessing for validation and testing sets (only rescaling)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using the generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels}),
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Flow validation images in batches using the generator
valid_generator = valid_test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': valid_files, 'class': valid_labels}),
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Flow testing images in batches using the generator
test_generator = valid_test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_files, 'class': test_labels}),
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator))

# Save the trained model
model.save('C:/Users/hp/Desktop/CV CP - Object Grip Detection/prosthetic_hand_grasp_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test accuracy:", test_accuracy)

# Predict the classes for test images and print them
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes for test images
true_classes = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Plot the test images along with their identified classes
plt.figure(figsize=(10, 10))
for i in range(len(test_files)):
    img = tf.keras.preprocessing.image.load_img(test_files[i], target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255
    ax = plt.subplot(len(test_files) // 5 + 1, 5, i + 1)
    plt.imshow(img_array[0])
    plt.title(f"Predicted: {class_labels[predicted_classes[i]]}\nTrue: {class_labels[true_classes[i]]}")
    plt.axis('off')
plt.show()

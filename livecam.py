import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define constants
IMAGE_SIZE = (224, 224)
model_path = "C:/Users/hp/Desktop/CV CP - Object Grip Detection/prosthetic_hand_grasp_model.h5"
class_labels = ['bottle', 'cup', 'mobile']  # Assuming these are the class labels used during training

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialize the video capture object for camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    img_array = np.expand_dims(resized_frame, axis=0) / 255

    # Predict the class of the frame
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    
    # Check if the predicted class index is within the range of class labels
    if predicted_class_idx < len(class_labels):
        predicted_class_label = class_labels[predicted_class_idx]
    else:
        predicted_class_label = "other"

    # Add predicted class label to the frame
    cv2.putText(frame, predicted_class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame from BGR to RGB (Matplotlib expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with predicted class label using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title(f"Predicted class: {predicted_class_label}")
    plt.show()

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

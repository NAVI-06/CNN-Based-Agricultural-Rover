import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('cotton.h5')


class_labels = ['Aphids', 'Army worm', 'Bacterial Blight','Healthy', 'Powdery Midew', 'Target spot']# Adjust based on your labels


IMG_HEIGHT, IMG_WIDTH = 150, 150


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))

    # Convert the image to RGB (OpenCV loads in BGR by default)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame (same as in the training pipeline)
    rgb_frame = rgb_frame / 255.0

    # Expand the dimensions of the frame to match model input (batch size, height, width, channels)
    frame_input = np.expand_dims(rgb_frame, axis=0)
    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get the class name based on the predicted index
    predicted_label = class_labels[predicted_class]

    # Display the resulting frame with the prediction label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Cotton Leaf Disease Detection', frame)
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

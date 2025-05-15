import cv2
import urllib.request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Global counters
global tar_count, hel_count, arm_count, bac_count, pow_count, aph_count
tar_count = hel_count = arm_count = bac_count = pow_count = aph_count = 0

# Load model and labels
model = tf.keras.models.load_model('cotton.h5')
class_labels = ['Healthy', 'Army worm', 'Bacterial Blight', 'Aphids', 'Powdery Midew', 'Target spot']
IMG_HEIGHT, IMG_WIDTH = 150, 150

# URL of IP camera
url = 'http://192.168.186.121/cam-hi.jpg'
cap = cv2.VideoCapture(url)


# Check webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = rgb_frame / 255.0
    frame_input = np.expand_dims(rgb_frame, axis=0)
    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class]

    # Count class occurrences
    if predicted_label == "Healthy":
        hel_count += 1
    elif predicted_label == "Army worm":
        arm_count += 1
    elif predicted_label == "Bacterial Blight":
        bac_count += 1
    elif predicted_label == "Powdery Midew":
        pow_count += 1
    elif predicted_label == "Target spot":
        tar_count += 1
    elif predicted_label == "Aphids":
        aph_count += 1

    # Display prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    if confidence > 0.7:
        cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.2f}%)", (10, 30), font, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Prediction: Uncertain", (10, 30), font, 1, (0, 0, 255), 2)

    cv2.imshow('Cotton Leaf Disease Detection', frame)

    # Print counts in terminal
    print("Aphids =", aph_count)
    print("Powdery Midew =", pow_count)
    print("Army worm =", arm_count)
    print("Healthy =", hel_count)
    print("Bacterial Blight =", bac_count)
    print("Target spot =", tar_count)



    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
labels = ['Healthy', 'Army worm', 'Bacterial Blight', 'Aphids', 'Powdery Midew', 'Target spot']
counts = [hel_count, arm_count, bac_count, aph_count, pow_count, tar_count]
colors = ['green', 'orange', 'grey', 'pink', 'red', 'blue']

plt.figure(figsize=(7, 7))
plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Final Disease Distribution After Detection')
plt.show()

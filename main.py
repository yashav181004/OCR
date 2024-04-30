import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))  # Resize to desired dimensions
    normalized = resized / 255.0  # Normalize pixel values
    return np.expand_dims(normalized, axis=-1)  # Add channel dimension for CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes is the number of output classes (characters)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
def predict_text(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(np.array([preprocessed_image]))
    predicted_label = np.argmax(prediction)
    return chr(predicted_label + ord('A'))  # Convert numeric label to character

image_path = 'path_to_image.jpg'
predicted_text = predict_text(image_path)
print(f'Predicted text: {predicted_text}')

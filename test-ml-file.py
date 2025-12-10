import cv2
import numpy as np
import tensorflow as tf
import os

folder_path = r'****'
model_path = 'model_emnist.h5'

def prepare_image(filepath):
    img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)
    if img is None: return None
    if len(img.shape) > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return simple_resize(img)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    digit_roi = img[y:y + h, x:x + w]
    h_roi, w_roi = digit_roi.shape
    aspect_ratio = h_roi / w_roi

    if aspect_ratio > 2.0:
        target_w = int(h_roi / 1.5)
        pad_w = (target_w - w_roi) // 2
        digit_roi = cv2.copyMakeBorder(digit_roi, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        h_roi, w_roi = digit_roi.shape

    final_img = np.zeros((28, 28), dtype=np.uint8)

    if h_roi > w_roi:
        scale = 20.0 / h_roi
        new_h, new_w = 20, int(w_roi * scale)
    else:
        scale = 20.0 / w_roi
        new_h, new_w = int(h_roi * scale), 20

    if new_w > 0 and new_h > 0:
        resized_roi = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        final_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_roi

    final_img = final_img.astype('float32') / 255.0
    final_img = np.expand_dims(final_img, axis=0)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img

def simple_resize(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def analyze_folder(folder, model_file):
    if not os.path.exists(model_file):
        print("Model not found.")
        return

    model = tf.keras.models.load_model(model_file)
    digit_counts = [0] * 10
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    files.sort()

    for filename in files:
        filepath = os.path.join(folder, filename)
        processed_img = prepare_image(filepath)

        if processed_img is not None:
            predictions = model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions)

            digit_counts[predicted_digit] += 1
            print(f"Analyzed {filename} -> {predicted_digit} (Confidence: {confidence:.2f})")
    return digit_counts

if __name__ == "__main__":
    if os.path.isdir(folder_path):
        result = analyze_folder_v3(folder_path, model_path)
        if result:
            print("Result:", result)
            print("Sum:", sum(result))

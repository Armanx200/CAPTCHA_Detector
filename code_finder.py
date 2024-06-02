import cv2
import numpy as np
from keras.models import load_model
import string

# Load the trained model
model = load_model('full_model.h5')  # Load your trained model file here

# Define symbols
symbols = string.ascii_lowercase + "0123456789"
num_symbols = len(symbols)

def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Image not detected")
        return ""

    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, num_symbols))
    l_ind = []
    for a in ans:
        l_ind.append(np.argmax(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt

# Example usage
captcha_image_path = input('Path to your captcha image')  # Path to your captcha image
predicted_text = predict(captcha_image_path)
print("Predicted Captcha:", predicted_text)

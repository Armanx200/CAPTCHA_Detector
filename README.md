```markdown
# DeepCaptcha Solver

A deep learning model for solving CAPTCHA images using convolutional neural networks (CNNs) built with Keras. 🤖🚀

![GitHub last commit](https://img.shields.io/github/last-commit/Armanx200/deepcaptcha-solver) ![GitHub stars](https://img.shields.io/github/stars/Armanx200/deepcaptcha-solver?style=social) ![GitHub forks](https://img.shields.io/github/forks/Armanx200/deepcaptcha-solver?style=social)

## Features 🌟

✨ Recognizes CAPTCHA images with high accuracy  
✨ Multi-symbol prediction for each letter  
✨ Trained on diverse CAPTCHA samples  

## Project Structure 📂

```
project_dir/
├── Samples/
├── code_finder.py
├── full_model.h5
├── model.py
└── model_weights.weights.h5
```

## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/Armanx200/deepcaptcha-solver.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀

### Training the Model 🏋️‍♂️

1. Run `model.py` to train the CAPTCHA recognition model.
2. Trained model weights will be saved as `model_weights.weights.h5`.
3. Save the entire model (architecture + weights) as `full_model.h5`.

### Captcha Code Recognition 🔍

1. Use `code_finder.py` to recognize CAPTCHA images.
2. Provide the path to the CAPTCHA image when prompted.
3. Get the predicted CAPTCHA code.

## Example 💡

```python
from code_finder import predict

captcha_image_path = input('🖼️ Path to your captcha image: ')
predicted_text = predict(captcha_image_path)
print("🔮 Predicted Captcha:", predicted_text)
```

## Acknowledgements 🙏

Special thanks to [Keras](https://keras.io/) and the CAPTCHA dataset providers.

## Contributing 🤝

Contributions are welcome! Feel free to open issues or pull requests.

## Author 👨‍💻

Armanx200 - [GitHub Profile](https://github.com/Armanx200)


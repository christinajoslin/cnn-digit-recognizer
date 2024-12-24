# MNIST Digit Recognition using CNN
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The project includes data preprocessing, model training, evaluation, and saving the trained model.
The trained model achieves an impressive test set accuracy of 99.0%, showcasing its effectiveness in digit recognition.

## Features
- Data preprocessing, including normalization and channel dimension addition.
- CNN architecture with two convolutional layers, max-pooling, and dense layers.
- Evaluation of the model's performance on test data.
- Saving the trained model for future use.

## Instructions
1. Ensure you have the required libraries installed (`tensorflow`, `scikit-learn`).
2. Run the notebook sequentially:
   - Load and preprocess the MNIST dataset.
   - Train the CNN model using the training and validation sets. (approx. 35-40 minute runtime)
   - Evaluate the model on the test set to assess accuracy and loss.
   - Save the trained model as `model_mnist.keras`.
3. Download the `model_mnist.keras` file for reuse.

## Dependencies
- **TensorFlow:** For building and training the CNN.
- **scikit-learn:** For splitting the dataset into training and validation sets.
- **Colab Environment:** For running the notebook.

## Notes
- Pixel values are normalized to the range [0, 1] for better performance.
- `random_state = 42` ensures reproducibility during dataset splitting.
- The trained model can be loaded using TensorFlow's `load_model` method.

## Author
Christina Joslin 

## Acknowledgements
- Data provided by the MNIST dataset.
- Thanks to the TensorFlow and scikit-learn teams for their open-source contributions.
"""

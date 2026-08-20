# AI-Powered Brain Tumor Classification

## About the Project

This project is an AI-based application that analyzes brain MRI images and classifies them into:

- No Tumor
- Benign Tumor
- Malignant Tumor

The application also provides an AI chatbot that can answer questions related to the diagnosis.

## Technologies Used

- Python
- PyTorch
- ResNet50
- Gradio
- Groq API
- Llama 3.1
- SQLite
- Requests
- Pillow

## How the Project Works

1. The user enters patient information.
2. The user uploads a brain MRI image.
3. The image is resized to 224 × 224 pixels and normalized.
4. The ResNet50 model analyzes the MRI image.
5. The model predicts whether the scan shows no tumor, a benign tumor, or a malignant tumor.
6. The diagnosis is displayed in the Gradio interface.
7. The user can ask a question related to the diagnosis.
8. Groq's Llama model generates an AI-based response.
9. The session details are saved in an SQLite database.
10. A summary can be sent to the patient's email through an email webhook.

## Main Features

- Brain MRI image classification
- Tumor classification into three categories
- AI-based medical question answering
- Patient session history
- SQLite database storage
- Email summary
- Simple Gradio web interface

## Project Files

- `app.py` - Main application code
- `requirements.txt` - Required Python libraries
- `interface.png` - Screenshot of the application interface
- `tumor_classifier_model.pth` - Trained model used by the application (not included due to file size)
- `README.md` - Project documentation

## Disclaimer

This project is intended for educational and demonstration purposes only. The AI-generated results should not be considered a medical diagnosis. A qualified medical professional should always be consulted for medical advice.

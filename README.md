# 🩺 Pneumonia Detection using Deep Learning (CNN)

This AI project uses Convolutional Neural Networks (CNN) to detect **Pneumonia from chest X-ray images**. It is built using TensorFlow/Keras and achieves high accuracy on the standard dataset.

---

## 📌 Features

- 📊 Classifies chest X-rays into:
  - `PNEUMONIA`
  - `NORMAL`
- 🧠 Trained using CNN (Convolutional Neural Network)
- 📁 Dataset: Chest X-Ray from Kaggle
- 🧪 Achieves high validation accuracy (~92-95%)
- 🖼️ Can test with new X-ray images

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV or PIL
- NumPy
- Matplotlib (for plotting)

---

## 🗂️ Folder Structure

Pneumonia_AI_Project/
├── model/ # Saved CNN model (.h5)
├── data/ # Train/test images (optional)
├── Pneumonia_CNN.ipynb # Main Jupyter notebook
├── test_image.jpg # Example X-ray image
└── README.md # You're here!

yaml
Copy
Edit

---

## 🚀 How to Run

1. 📦 Install dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
▶️ Run the notebook:

bash
Copy
Edit
jupyter notebook Pneumonia_CNN.ipynb
🔍 Predict on custom image:

python
Copy
Edit
model.predict(preprocess("test_image.jpg"))
📉 Sample Results
Metric	Value
Accuracy	94.7%
Loss	Low
Model Used	CNN (3 Conv layers + Dense)

📸 Example Output


📚 Dataset Source
Kaggle: Chest X-Ray Images (Pneumonia)

🙌 Team
👨‍💻 Sagar (Lead Developer)
👨‍💻 Ankur
👨‍💻 Soumyaranjan

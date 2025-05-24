# 🧠 Clothing Similarity Search using Siamese Neural Network

A computer vision project that uses a **Siamese Neural Network** to retrieve visually similar clothing items, such as jackets, jeans, shirts, and shoes, from a custom image database. Built with TensorFlow and deployed via a Flask web app.

## 📌 Project Overview

This system allows users to upload a clothing image and receive similar items from a predefined image database, along with metadata (e.g., product info or price). It uses one-shot learning to determine similarity between pairs of images, inspired by the paper:

> Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition.

---

## 🚀 How to Run the Project

Follow these steps to run the clothing similarity search system on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/amromeshref/Computer-Vision-Project.git
cd Computer-Vision-Project
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Start the Flask server
```bash
python main.py
```

### 4. Open your browser and navigate to
```
http://localhost:5000
```

### 5. Upload an image and choose a category, and then you can see the similar images and their info displayed on the page!

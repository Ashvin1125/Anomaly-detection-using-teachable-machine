# ⚙️ Casting Defect Detection using Deep Learning

A Streamlit web application for real-time anomaly detection in manufactured cast metal parts. This project leverages a Convolutional Neural Network (CNN) to classify product images as 'Normal' or 'Anomalous' (defective), helping to automate the quality control process in a manufacturing setting.

  
*(**Action:** Replace the image link above with a screenshot of your own running app\!)*

-----

## 📋 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [Features](https://www.google.com/search?q=%23-features)
  - [Technology Stack](https://www.google.com/search?q=%23-technology-stack)
  - [Project Structure](https://www.google.com/search?q=%23-project-structure)
  - [Setup and Installation](https://www.google.com/search?q=%23-setup-and-installation)
  - [Deployment](https://www.google.com/search?q=%23-deployment)
  - [Dataset](https://www.google.com/search?q=%23-dataset)
  - [License](https://www.google.com/search?q=%23-license)

-----

## 📝 Project Overview

In manufacturing, ensuring product quality is critical. Manual inspection is often slow, expensive, and prone to human error. This project presents a proof-of-concept for an automated quality control system. It uses a deep learning model to analyze images of cast metal impellers and identify those with manufacturing defects. The entire system is wrapped in a user-friendly web interface built with Streamlit.

-----

## ✨ Features

  * **Image-Based Detection:** Users can upload an image of a cast part for immediate analysis.
  * **Real-time Classification:** The app classifies the part as `Normal` or `Anomaly` using a trained TensorFlow/Keras model.
  * **Confidence Scoring:** Displays the model's confidence in its prediction, providing a measure of certainty.
  * **Actionable Recommendations:** Provides clear, color-coded results and a practical recommendation (e.g., "Flag for manual inspection").
  * **Informational Interface:** Includes a sidebar with project details and an expandable "How-to-Use" guide.
  * **Example Cases:** Shows example images of normal and defective parts for user reference.

-----

## 🛠️ Technology Stack

  * **Model Training:** Google's Teachable Machine
  * **Deep Learning Framework:** TensorFlow / Keras 3
  * **Web Application Framework:** Streamlit
  * **Core Language:** Python 3.10
  * **Key Libraries:** NumPy, Pillow, OpenCV-Python
  * **Deployment:** Streamlit Community Cloud

-----

## 📂 Project Structure

The project repository is organized as follows:

```
casting-defect-detector/
│
├── my_model/                 # Exported TensorFlow SavedModel folder
│   ├── saved_model.pb
│   └── ... (variables & assets)
│
├── app.py                    # The main Streamlit application script
├── requirements.txt          # Python dependencies for deployment
├── labels.txt                # Class labels for the model (Normal, Anomaly)
├── ok_example.jpeg           # Example image of a normal part
├── def_example.jpeg          # Example image of a defective part
└── README.md                 # This file
```

-----

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [Your GitHub Repository URL]
    cd casting-defect-detector
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    The application will open in your web browser.

-----

## ☁️ Deployment

The application is deployed on Streamlit Community Cloud and is publicly accessible.

**Live App URL:** [**Your Live Streamlit App URL Here**]

-----

## 📊 Dataset

The model was trained on the **Casting Product Image Data for Quality Inspection** dataset, which is publicly available on Kaggle. This dataset contains a collection of images for both defective and okay cast products.

  * **Dataset Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

-----


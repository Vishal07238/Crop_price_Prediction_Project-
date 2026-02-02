Markdown
# Soyabean Price Prediction System 🌾

## Overview
This is a Machine Learning web application designed to forecast Soyabean prices for various APMC (Agricultural Produce Market Committee) markets in Maharashtra. The system helps farmers and traders make informed decisions by predicting future prices based on historical data.

The application uses a **Flask** backend to serve a machine learning model and a responsive **HTML/CSS** frontend for user interaction.

## 🚀 Features
- **Market Selection:** Dropdown menu to select specific APMC markets (e.g., Nagpur, Nanded).
- **Price Input:** Interface to input the last 7 days of market prices.
- **Forecasting:** Generates predictions for:
  - **3-Day Forecast:** Short-term price trend.
  - **7-Day Forecast:** Weekly price trend.
- **Interactive UI:** Clean and responsive design using modern CSS.

## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, NumPy, Pandas
- **Data Source:** Historical APMC price data (`maharashtra_soyabean_data.xlsx`)

## 📂 Project Structure
```text
/Soyabean-Project
│
├── app.py                   # Main Flask application (Backend)
├── model_training.ipynb     # Jupyter Notebook used to train the model
├── requirements.txt         # List of Python dependencies
├── maharashtra_soyabean_data.xlsx  # Dataset used for training
├── soyabean_model_files/    # Folder containing trained .pkl model files
├── static/                  # CSS, Images, and JavaScript files
└── templates/               # HTML files (index.html)
⚙️ How to Run Locally
Clone the Repository

Bash
git clone https://github.com/Vishal07238/Crop_price_Prediction_Project-.git
cd Soyabean-Price-Predictor
Create a Virtual Environment (Optional but Recommended)

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application

Bash
python app.py
Access the App Open your browser and go to: http://127.0.0.1:5000/

📊 Model Information
The model was trained using historical data from Maharashtra APMC markets. It analyzes trends from the past 7 days to predict future price movements. You can view the training process in the model_training.ipynb file.

📸 Screenshots
(Add your screenshots here. You can drag and drop images into the GitHub issue or README editor)

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request.
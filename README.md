# Azure Cost Prediction ANN

## Project Overview
This project predicts Azure cloud service costs using an Artificial Neural Network (ANN).

## Dataset
Features:
- UsageDate
- ServiceName
- CostUSD
- Currency

Target:
- Cost

## Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Flask
- Pandas
- NumPy

## Project Structure
```
Azure-Cost-Prediction-ANN/
│── app.py
│── model.keras
│── preprocessor.pkl
│── requirements.txt
│── README.md
│── templates/
│── static/
```

## Installation
```bash
pip install -r requirements.txt
```

## Run the Application
```bash
python app.py
```

Open:
http://127.0.0.1:5000/

## API Input Example
```json
{
  "UsageDate":"2023-05-15",
  "ServiceName":"Storage",
  "CostUSD":120.75,
  "Currency":"INR"
}
```

## Output
Returns the predicted Azure service cost.

## Author
Reddy

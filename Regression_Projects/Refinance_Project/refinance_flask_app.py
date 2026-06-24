from flask import Flask, render_template, request
import pandas as pd
from refinance_model_train_and_predict import predict  # Import your predict function

app = Flask(__name__)

feature_columns = [
    'Month No', 'Mortgage Rate', 'Inflation', 'Housing Price Index', 'Treasury Yield',
    'Unemployment Rate', 'GDP', 'Business Confidence Index', 'Consumer Confidence Index',
    'Initial Unemployment Claim', 'Disposable Income'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(col, 0.0)) for col in feature_columns]
            input_df = pd.DataFrame([input_data], columns=feature_columns)
            prediction = predict(input_df)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', feature_columns=feature_columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from Utils import preprocess_new


# Initialize
app = Flask(__name__)

model = joblib.load('Model_XGB.pkl')

# Home


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

# Predict


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Long = float(request.form['Long'])
        Latit = float(request.form['Latit'])
        Housing_median = float(request.form['Housing_median'])
        Tot_Rooms = float(request.form['Tot_Rooms'])
        Tot_Bedrooms = float(request.form['Tot_Bedrooms'])
        Pop = float(request.form['Pop'])
        holds = float(request.form['holds'])
        Med_Income = float(request.form['Med_Income'])
        Ocean_Prox = request.form['Ocean_Prox']
        rooms_per_hold = Tot_Rooms / holds
        bedrooms_per_rooms = Tot_Bedrooms / Tot_Rooms
        pop_per_hold = Pop / holds

        X_new = pd.DataFrame({'longitude': [Long], 'latitude': [Latit], 'housing_median_age': [Housing_median],
                              'total_rooms': [Tot_Rooms], 'total_bedrooms': [Tot_Bedrooms], 'population': [Pop],
                              'households': [holds], 'median_income': [Med_Income], 'ocean_proximity': [Ocean_Prox],
                              'rooms_per_household': [rooms_per_hold], 'bedroms_per_rooms': [bedrooms_per_rooms], 'population_per_household': [pop_per_hold]})

        X_proccessed = preprocess_new(X_new)

        y_pred_new = model.predict(X_proccessed)
        y_pred_new = '{:.4f} $'.format(y_pred_new[0])

        return render_template('predict.html', pred_vals=y_pred_new)

    else:
        return render_template('predict.html')


# About
@app.route('/about')
def about():
    return render_template('about.html')


# Terminal
if __name__ == '__main__':
    app.run(debug=True)

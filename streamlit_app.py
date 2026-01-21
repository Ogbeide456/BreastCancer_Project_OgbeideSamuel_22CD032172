
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load(os.path.join('model', 'breast_cancer_model.pkl'))

FEATURES = ['radius_mean','perimeter_mean','area_mean','smoothness_mean','concavity_mean']

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        values = [float(request.form[f]) for f in FEATURES]
        pred = model.predict([values])[0]
        result = "Malignant" if pred == 1 else "Benign"
    return render_template('index.html', features=FEATURES, result=result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)
# loading models
etr = pickle.load(open('etr (1).pkl', 'rb'))
scaler = pickle.load(open('scaler (1).pkl', 'rb'))

# flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        yearFrom = request.form['Yearfrom']
        yearTo = request.form['YearTo']
        month = request.form['month']
        fort_night = request.form['fort_night']
        CLR_Incidence = request.form['CLR']
        T_max = request.form['Temp_max']
        T_min = request.form['Temp_min']
        RH = request.form['RH']
        RF = request.form['RF']
        OC = request.form['OC']
        P = request.form['P']
        K = request.form['K']
        pH = request.form['pH']
        spacing = request.form['Spacing']
        Age = request.form['Age']

        features = np.array(
            [[yearFrom, yearTo, month, fort_night, CLR_Incidence, T_max, T_min, RH, RF, OC, P, K, pH, spacing, Age]],
            dtype=object)
        transform_features = scaler.transform(features)
        prediction = etr.predict(transform_features).reshape(-1, 1)

        return render_template('index.html', prediction=prediction[0][0])


if __name__ == "__main__":
    app.run(debug=True)

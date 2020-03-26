from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction=output)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask("_name__")

rf = pickle.load(open('crop.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods = ['POST','GET'])

def predict():
    n = (request.form['n'])
    pho = (request.form['p'])
    k = (request.form['k'])
    te = (request.form['temp'])
    hum = (request.form['hum'])
    p = (request.form['ph'])
    ra = (request.form['rain'])
    values = np.array([[n, pho, k, te, hum, p, ra]])

    # int_features=[int(x) for x in request.form.values()]
    # values=[np.array(int_features)]
    # values = np.array([[99,15,27,27.41,56.6,6.08,127.92]])

    output=rf.predict(values)
    # print(predict)

    return render_template('result.html',predict=output)


if __name__ == '__main__':
    app.run(debug=True)
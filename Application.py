import pickle
import numpy as np

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModelHouse.pk1", "rb"))
house = pd.read_csv("Cleaned_HouseData.csv")

@app.route('/')
def index():
    cities = sorted(house['City'].unique())
    area = sorted(house['Area'].unique())
    no_of_rooms = sorted(house['No. of Bedrooms'].unique())
    resale = house['Resale'].unique()

    return render_template('index.html', cities=cities, area=area, no_of_rooms=no_of_rooms, resale=resale, )

@app.route('/predict',methods=['POST'])

def predict():

    cities = request.form.get('city')

    area = request.form.get('area')
    no_of_rooms = request.form.get('no_of_rooms')
    resale = request.form.get('resale')


    prediction = model.predict(pd.DataFrame(columns=['City', 'Area', 'No. of Bedrooms', 'Resale'],
                              data = np.array([cities, area, no_of_rooms, resale]).reshape(1, 4)))
    print(prediction)

    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)

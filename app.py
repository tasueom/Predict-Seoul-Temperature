from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

app = Flask(__name__)

model = LinearRegression()
df = pd.read_csv('2024년11월_서울시기온.csv')
df = df.dropna(inplace=False).reset_index(drop=True)
model.fit(df[['일시']], df['평균기온'])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
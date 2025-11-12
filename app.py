from flask import Flask, render_template, request, json
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

model = LinearRegression()
# CSV 파일을 불러와 DataFrame 생성
df = pd.read_csv('2024년11월_서울시기온.csv')
# 결측 행 제거 후 인덱스를 0부터 다시 설정
df = df.dropna().reset_index(drop=True)
# 문자열로 읽힌 날짜를 pandas Timestamp로 변환
df['일시'] = pd.to_datetime(df['일시'])
# 각 날짜를 선형 회귀 입력으로 쓰기 위해 서수(ordinal) 정수로 변환
df['일시_ordinal'] = df['일시'].map(pd.Timestamp.toordinal)
# 날짜(서수)와 평균기온 간의 관계를 학습
model.fit(df[['일시_ordinal']], df['평균기온'])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
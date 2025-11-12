from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import json

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

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    # 문자열 입력을 Timestamp로 변환하여 날짜 연산이 가능하도록 준비
    start_date = pd.to_datetime(date_str)
    predictions = []
    for offset in range(7):
        # pd.Timedelta로 기준 날짜에서 offset일 만큼 이동
        target_date = start_date + pd.Timedelta(days=offset)
        # 날짜를 서수 정수로 바꿔 모델 입력 형태에 맞춤
        prediction_value = model.predict([[target_date.toordinal()]])
        predictions.append(
            {
                # Chart.js에서 x축 라벨로 사용할 날짜 문자열
                'date': target_date.strftime('%Y-%m-%d'),
                # Chart.js에서 y값으로 사용할 평균 기온 (float으로 변환해 JSON 직렬화 대비)
                'temperature': float(prediction_value[0]),
            }
        )
    # Chart.js에서 바로 사용할 수 있도록 JSON으로 반환
    data = json.dumps(predictions, ensure_ascii=False)
    return render_template('result.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
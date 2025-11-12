from flask import Flask, render_template, request, send_file, abort
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
from io import BytesIO
from io import BytesIO

app = Flask(__name__)

model = LinearRegression()
# 다년간의 기온 데이터 로드
df = pd.read_csv('서울시기온.csv')
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
    target_dates = [start_date + pd.Timedelta(days=offset) for offset in range(7)]
    labels = [target_date.strftime('%m-%d') for target_date in target_dates]

    predictions = []
    for target_date in target_dates:
        # 날짜를 서수 정수로 바꿔 모델 입력 형태에 맞춤
        prediction_value = model.predict([[target_date.toordinal()]])
        predictions.append(
            {
                # Chart.js에서 x축 라벨로 사용할 날짜 문자열
                'date': target_date.strftime('%m-%d'),
                # Chart.js에서 y값으로 사용할 평균 기온 (float으로 변환해 JSON 직렬화 대비)
                'temperature': float(prediction_value[0]),
            }
        )
    # Chart.js 데이터셋 구성
    datasets = [
        {
            'label': f"{start_date.year} 예측",
            'data': [item['temperature'] for item in predictions],
            'borderColor': 'rgba(255, 99, 132, 1)',
            'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            'tension': 0.2,
            'fill': False,
        }
    ]

    palette = [
        ('rgba(54, 162, 235, 1)', 'rgba(54, 162, 235, 0.2)'),
        ('rgba(255, 206, 86, 1)', 'rgba(255, 206, 86, 0.2)'),
        ('rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 0.2)'),
        ('rgba(153, 102, 255, 1)', 'rgba(153, 102, 255, 0.2)'),
        ('rgba(255, 159, 64, 1)', 'rgba(255, 159, 64, 0.2)'),
    ]

    unique_years = sorted(df['일시'].dt.year.unique(), reverse=True)
    color_index = 0
    for year in unique_years:
        actuals = []
        for target_date in target_dates:
            mask = (
                (df['일시'].dt.year == year)
                & (df['일시'].dt.month == target_date.month)
                & (df['일시'].dt.day == target_date.day)
            )
            if not mask.any():
                actuals.append(None)
            else:
                actuals.append(float(df.loc[mask, '평균기온'].iloc[0]))

        if all(value is None for value in actuals):
            continue

        border_color, background_color = palette[color_index % len(palette)]
        color_index += 1

        datasets.append(
            {
                'label': f"{year} 실제",
                'data': actuals,
                'borderColor': border_color,
                'backgroundColor': background_color,
                'tension': 0.2,
                'fill': False,
                'spanGaps': True,
            }
        )

    chart_payload = {
        'labels': labels,
        'datasets': datasets,
    }

    # Chart.js에서 바로 사용할 수 있도록 JSON으로 반환
    data = json.dumps(chart_payload, ensure_ascii=False)
    start_date_str = start_date.strftime('%Y-%m-%d')
    return render_template('result.html', data=data, start_date=start_date_str)


@app.route('/export')
def export_predictions():
    date_str = request.args.get('date')
    if not date_str:
        abort(400, description='date 파라미터가 필요합니다.')

    try:
        start_date = pd.to_datetime(date_str)
    except (ValueError, TypeError):
        abort(400, description='잘못된 날짜 형식입니다.')

    records = []
    for offset in range(7):
        target_date = start_date + pd.Timedelta(days=offset)
        prediction_value = model.predict([[target_date.toordinal()]])
        records.append(
            {
                '날짜': target_date.strftime('%Y-%m-%d'),
                '예상온도': float(prediction_value[0]),
            }
        )

    if not records:
        abort(404, description='내보낼 예측 데이터가 없습니다.')

    df_export = pd.DataFrame(records)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False)
    output.seek(0)

    filename = f"temperature_forecast_{start_date.strftime('%Y%m%d')}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

@app.route('/export', methods=['POST'])
def export():
    buffer = BytesIO()

if __name__ == '__main__':
    app.run(debug=True)
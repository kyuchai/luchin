# 台股成交金額分析

Streamlit app for:
- 個股：TWSE 上市 / TPEx 上櫃
- 大盤：臺灣加權股價指數
- 今日 / 指定單日 / 日期區間
- 成交金額價差比 = 成交金額（億） / (最高 - 最低)
- 區間平均 x3 ALERT

## Local

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Render

此專案已包含 `render.yaml`。

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

建議 Python 3.12。

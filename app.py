import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# 1. 基本設定
# ============================================================

TZ = ZoneInfo("Asia/Taipei")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


# ============================================================
# 2. API
# ============================================================

TWSE_MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
TWSE_STOCK_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
TPEX_STOCK_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
TPEX_OPENAPI_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TAIEX_HISTORY_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
MARKET_TURNOVER_URL = "https://www.twse.com.tw/exchangeReport/FMTQIK"


# ============================================================
# 3. 市場設定
# ============================================================

MARKET_OVERRIDES = {
    "6770": "TW",
}

MARKET_LABEL = {
    "TW": "上市",
    "TWO": "上櫃",
}


# ============================================================
# 4. Streamlit
# ============================================================

st.set_page_config(
    page_title="台股成交金額分析",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 5. 共用函數
# ============================================================

def now_taipei():
    return datetime.now(TZ)


def clean_number(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if value in {"", "-", "--", "---", "N/A", "nan", "None"}:
        return np.nan

    value = (
        value
        .replace(",", "")
        .replace("X", "")
        .replace("+", "")
    )

    try:
        return float(value)
    except Exception:
        return np.nan


def roc_to_datetime(value):
    try:
        parts = str(value).strip().split("/")

        if len(parts) != 3:
            return pd.NaT

        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])

        return pd.Timestamp(
            year=year,
            month=month,
            day=day
        )

    except Exception:
        return pd.NaT


# ============================================================
# 6. 顯示格式
# ============================================================

def format_price(value):
    if pd.isna(value):
        return "-"
    return f"{value:,.2f}"


def format_billion(value):
    if pd.isna(value):
        return "-"
    return f"{value:,.2f} 億"


# ============================================================
# 7. 計算
# ============================================================

def add_calculation_columns(df):
    df = df.copy()

    df["價差"] = (
        df["最高價"]
        -
        df["最低價"]
    )

    df["成交金額(億)"] = (
        df["成交金額"]
        /
        100_000_000
    )

    df["成交金額價差比(億)"] = np.where(
        df["價差"] > 0,
        (
            df["成交金額"]
            /
            df["價差"]
            /
            100_000_000
        ),
        np.nan
    )

    return df


# ============================================================
# 8. 個股 MIS
# ============================================================

def get_mis_snapshot(stock_id, market):
    prefix = "tse" if market == "TW" else "otc"

    params = {
        "ex_ch": f"{prefix}_{stock_id}.tw",
        "json": "1",
        "delay": "0",
        "_": int(time.time() * 1000),
    }

    try:
        response = requests.get(
            TWSE_MIS_URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        rows = data.get("msgArray", [])

        if not rows:
            return None

        x = rows[0]

        code = str(x.get("c", "")).strip()

        if code != stock_id:
            return None

        stock_name = (
            str(x.get("n", "")).strip()
            or str(x.get("nf", "")).strip()
            or stock_id
        )

        date_text = str(x.get("d", "")).strip()

        try:
            quote_date = pd.to_datetime(
                date_text,
                format="%Y%m%d"
            )
        except Exception:
            quote_date = pd.Timestamp(
                now_taipei().date()
            )

        return {
            "股票代號": stock_id,
            "股票名稱": stock_name,
            "日期": quote_date,
            "時間": str(x.get("t", "")).strip(),
            "最高價": clean_number(x.get("h")),
            "最低價": clean_number(x.get("l")),
            "目前價": clean_number(x.get("z")),
        }

    except Exception:
        return None


# ============================================================
# 8-1. TPEx OpenAPI：上櫃最新交易日資料
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_tpex_latest_openapi(stock_id):
    """
    使用 TPEx 官方 OpenAPI 辨識上櫃股票，並取得最新交易日行情。
    這個端點不需要 API Key，且比網頁型歷史介面更適合 Render。
    """
    try:
        response = requests.get(
            TPEX_OPENAPI_URL,
            timeout=20,
            headers={
                "User-Agent": HEADERS["User-Agent"],
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        rows = response.json()

        if not isinstance(rows, list):
            return None

        for row in rows:
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if code != stock_id:
                continue

            date_text = str(row.get("Date", "")).strip()
            trade_date = pd.NaT

            # TPEx OpenAPI 日期通常為民國年月日，例如 1150821
            try:
                if len(date_text) >= 7 and date_text.isdigit():
                    roc_year = int(date_text[:-4])
                    month = int(date_text[-4:-2])
                    day = int(date_text[-2:])
                    trade_date = pd.Timestamp(
                        year=roc_year + 1911,
                        month=month,
                        day=day,
                    )
            except Exception:
                trade_date = pd.NaT

            return {
                "股票代號": stock_id,
                "股票名稱": str(row.get("CompanyName", stock_id)).strip() or stock_id,
                "日期": trade_date,
                "時間": "",
                "開盤價": clean_number(row.get("Open")),
                "最高價": clean_number(row.get("High")),
                "最低價": clean_number(row.get("Low")),
                "目前價": clean_number(row.get("Close")),
                "收盤價": clean_number(row.get("Close")),
                "成交股數": clean_number(row.get("TradingShares")),
                "成交金額": clean_number(row.get("TransactionAmount")),
                "成交筆數": clean_number(row.get("TransactionNumber")),
                "資料來源": "TPEx OpenAPI",
            }

        return None

    except Exception:
        return None


def tpex_openapi_to_snapshot(latest):
    if not latest:
        return None

    return {
        "股票代號": latest["股票代號"],
        "股票名稱": latest["股票名稱"],
        "日期": latest["日期"],
        "時間": "",
        "最高價": latest["最高價"],
        "最低價": latest["最低價"],
        "目前價": latest["收盤價"],
    }


# ============================================================
# 9. 個股市場辨識
# ============================================================

def detect_market(stock_id):
    # 1. 已知特殊指定
    if stock_id in MARKET_OVERRIDES:
        market = MARKET_OVERRIDES[stock_id]
        return (
            market,
            get_mis_snapshot(stock_id, market)
        )

    # 2. 先查 TWSE MIS（上市）
    tw_snapshot = get_mis_snapshot(stock_id, "TW")
    if tw_snapshot:
        return "TW", tw_snapshot

    # 3. 再查 TPEx MIS（上櫃盤中）
    two_snapshot = get_mis_snapshot(stock_id, "TWO")
    if two_snapshot:
        return "TWO", two_snapshot

    # 4. TPEx 官方 OpenAPI：Render 上的主要上櫃辨識備援
    tpex_latest = get_tpex_latest_openapi(stock_id)
    if tpex_latest:
        return "TWO", tpex_openapi_to_snapshot(tpex_latest)

    # 5. Yahoo 最後備援
    for market, suffix in [
        ("TW", ".TW"),
        ("TWO", ".TWO")
    ]:
        try:
            df = yf.download(
                stock_id + suffix,
                period="5d",
                progress=False,
                auto_adjust=False,
                threads=False
            )

            if df is not None and not df.empty:
                return market, None

        except Exception:
            pass

    return None, None


# ============================================================
# 10. TWSE 個股月資料
# ============================================================

def get_twse_stock_month(stock_id, year, month):
    params = {
        "response": "json",
        "date": f"{year}{month:02d}01",
        "stockNo": stock_id
    }

    try:
        r = requests.get(
            TWSE_STOCK_URL,
            params=params,
            headers=HEADERS,
            timeout=15
        )

        r.raise_for_status()

        j = r.json()

        rows = j.get("data", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).iloc[:, :9]

        df.columns = [
            "日期",
            "成交股數",
            "成交金額",
            "開盤價",
            "最高價",
            "最低價",
            "收盤價",
            "漲跌價差",
            "成交筆數"
        ]

        df["日期"] = df["日期"].apply(roc_to_datetime)

        for col in df.columns[1:]:
            df[col] = df[col].apply(clean_number)

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# 11. TPEx 個股月資料
# ============================================================

def get_tpex_stock_month(stock_id, year, month):
    """
    TPEx 官方個股歷史月資料。
    網頁型介面有時會對雲端 IP 比較敏感，因此加入重試與較簡單 Header。
    若仍失敗，get_stock_range() 會再使用 Yahoo .TWO 歷史資料備援。
    """
    params = {
        "code": stock_id,
        "date": f"{year}/{month:02d}/01",
        "response": "json"
    }

    request_headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.tpex.org.tw/",
    }

    for attempt in range(3):
        try:
            r = requests.get(
                TPEX_STOCK_URL,
                params=params,
                headers=request_headers,
                timeout=20,
                allow_redirects=True,
            )

            r.raise_for_status()
            j = r.json()

            # 正常回傳 stat=ok 並有 tables
            if str(j.get("stat", "")).lower() not in {"ok", ""}:
                return pd.DataFrame()

            tables = j.get("tables", [])
            if not tables:
                return pd.DataFrame()

            rows = tables[0].get("data", [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).iloc[:, :9]
            df.columns = [
                "日期",
                "成交張數",
                "成交仟元",
                "開盤價",
                "最高價",
                "最低價",
                "收盤價",
                "漲跌價差",
                "成交筆數"
            ]

            df["日期"] = df["日期"].apply(roc_to_datetime)

            for col in df.columns[1:]:
                df[col] = df[col].apply(clean_number)

            # TPEx 原始欄位：成交張數、成交仟元
            df["成交股數"] = df["成交張數"] * 1000
            df["成交金額"] = df["成交仟元"] * 1000
            df["資料來源"] = "TPEx 官方歷史資料"

            return df[[
                "日期",
                "成交股數",
                "成交金額",
                "開盤價",
                "最高價",
                "最低價",
                "收盤價",
                "漲跌價差",
                "成交筆數",
                "資料來源",
            ]]

        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))

    return pd.DataFrame()


def get_yahoo_stock_range(stock_id, market, start_date, end_date):
    """
    歷史行情備援。
    Yahoo 不提供台股每日官方成交金額，因此以 typical price × volume 估算成交金額。
    只有官方 TPEx 歷史資料取得失敗時才會使用。
    """
    suffix = ".TW" if market == "TW" else ".TWO"

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        raw = yf.download(
            stock_id + suffix,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        raw.rename(columns={
            "Date": "日期",
            "Open": "開盤價",
            "High": "最高價",
            "Low": "最低價",
            "Close": "收盤價",
            "Volume": "成交股數",
        }, inplace=True)

        raw["日期"] = pd.to_datetime(raw["日期"]).dt.tz_localize(None)

        for col in ["開盤價", "最高價", "最低價", "收盤價", "成交股數"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        typical_price = (
            raw["最高價"] + raw["最低價"] + raw["收盤價"]
        ) / 3

        raw["成交金額"] = typical_price * raw["成交股數"]
        raw["漲跌價差"] = raw["收盤價"].diff()
        raw["成交筆數"] = np.nan
        raw["資料來源"] = "Yahoo 歷史備援（成交金額估算）"

        return raw[[
            "日期",
            "成交股數",
            "成交金額",
            "開盤價",
            "最高價",
            "最低價",
            "收盤價",
            "漲跌價差",
            "成交筆數",
            "資料來源",
        ]]

    except Exception:
        return pd.DataFrame()


# ============================================================
# 12. 個股歷史區間
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_stock_range(
    stock_id,
    market,
    start_date,
    end_date
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    months = pd.period_range(
        start=start,
        end=end,
        freq="M"
    )

    all_data = []

    for period in months:
        if market == "TW":
            df = get_twse_stock_month(
                stock_id,
                period.year,
                period.month
            )
        else:
            df = get_tpex_stock_month(
                stock_id,
                period.year,
                period.month
            )

            # Render / Cloudflare 若擋下 TPEx 歷史頁，改抓 Yahoo .TWO
            if df.empty:
                month_start = pd.Timestamp(
                    year=period.year,
                    month=period.month,
                    day=1
                )
                month_end = month_start + pd.offsets.MonthEnd(1)

                df = get_yahoo_stock_range(
                    stock_id,
                    market,
                    max(start, month_start),
                    min(end, month_end)
                )

        if not df.empty:
            all_data.append(df)

        time.sleep(0.08)

    if not all_data:
        # 最後整段再嘗試一次 Yahoo，避免月資料全數被 TPEx 擋下
        if market == "TWO":
            fallback = get_yahoo_stock_range(
                stock_id,
                market,
                start,
                end
            )

            if not fallback.empty:
                return add_calculation_columns(
                    fallback.sort_values("日期").reset_index(drop=True)
                )

        return pd.DataFrame()

    result = pd.concat(
        all_data,
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=["日期"],
        keep="last"
    )

    result = result[
        (result["日期"] >= start)
        &
        (result["日期"] <= end)
    ]

    result = (
        result
        .sort_values("日期")
        .reset_index(drop=True)
    )

    return add_calculation_columns(result)


# ============================================================
# 13. 大盤指數月 OHLC
# ============================================================

def get_taiex_month(year, month):
    params = {
        "response": "json",
        "date": f"{year}{month:02d}01"
    }

    try:
        r = requests.get(
            TAIEX_HISTORY_URL,
            params=params,
            headers=HEADERS,
            timeout=15
        )

        r.raise_for_status()

        j = r.json()

        rows = j.get("data", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).iloc[:, :5]

        df.columns = [
            "日期",
            "開盤價",
            "最高價",
            "最低價",
            "收盤價"
        ]

        df["日期"] = df["日期"].apply(
            roc_to_datetime
        )

        for col in [
            "開盤價",
            "最高價",
            "最低價",
            "收盤價"
        ]:
            df[col] = df[col].apply(
                clean_number
            )

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# 14. 大盤成交金額月資料
# ============================================================

def get_market_turnover_month(year, month):
    params = {
        "response": "json",
        "date": f"{year}{month:02d}01"
    }

    try:
        r = requests.get(
            MARKET_TURNOVER_URL,
            params=params,
            headers=HEADERS,
            timeout=15
        )

        r.raise_for_status()

        j = r.json()

        rows = j.get("data", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).iloc[:, :6]

        df.columns = [
            "日期",
            "成交股數",
            "成交金額",
            "成交筆數",
            "加權指數",
            "漲跌點數"
        ]

        df["日期"] = df["日期"].apply(
            roc_to_datetime
        )

        for col in df.columns[1:]:
            df[col] = df[col].apply(
                clean_number
            )

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# 15. 大盤歷史區間
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_index_range(
    start_date,
    end_date
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    months = pd.period_range(
        start=start,
        end=end,
        freq="M"
    )

    all_index = []
    all_turnover = []

    for period in months:
        idx = get_taiex_month(
            period.year,
            period.month
        )

        val = get_market_turnover_month(
            period.year,
            period.month
        )

        if not idx.empty:
            all_index.append(idx)

        if not val.empty:
            all_turnover.append(val)

        time.sleep(0.08)

    if not all_index:
        return pd.DataFrame()

    index_df = pd.concat(
        all_index,
        ignore_index=True
    )

    index_df = index_df.drop_duplicates(
        subset=["日期"]
    )

    if all_turnover:
        turnover_df = pd.concat(
            all_turnover,
            ignore_index=True
        )

        turnover_df = turnover_df.drop_duplicates(
            subset=["日期"]
        )

        index_df = pd.merge(
            index_df,
            turnover_df[[
                "日期",
                "成交金額"
            ]],
            on="日期",
            how="left"
        )
    else:
        index_df["成交金額"] = np.nan

    index_df = index_df[
        (index_df["日期"] >= start)
        &
        (index_df["日期"] <= end)
    ]

    index_df = (
        index_df
        .sort_values("日期")
        .reset_index(drop=True)
    )

    return add_calculation_columns(
        index_df
    )


# ============================================================
# 16. 個股盤中成交金額估算
# ============================================================

def get_stock_intraday(stock_id, market):
    suffix = ".TW" if market == "TW" else ".TWO"

    try:
        df = yf.download(
            stock_id + suffix,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return None

        high = pd.to_numeric(
            df["High"],
            errors="coerce"
        ).max()

        low = pd.to_numeric(
            df["Low"],
            errors="coerce"
        ).min()

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        volume = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        ).fillna(0)

        turnover = (
            close
            *
            volume
        ).sum()

        return {
            "最高價": high,
            "最低價": low,
            "成交金額": turnover
        }

    except Exception:
        return None


# ============================================================
# 17. 大盤今日指數
# ============================================================

def get_taiex_intraday():
    try:
        df = yf.download(
            "^TWII",
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return None

        return {
            "最高價":
                pd.to_numeric(
                    df["High"],
                    errors="coerce"
                ).max(),

            "最低價":
                pd.to_numeric(
                    df["Low"],
                    errors="coerce"
                ).min(),

            "收盤價":
                pd.to_numeric(
                    df["Close"],
                    errors="coerce"
                ).iloc[-1]
        }

    except Exception:
        return None


# ============================================================
# 18. 個股今日
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def get_today_stock(stock_id):
    market, snapshot = detect_market(stock_id)

    if market is None:
        raise ValueError(
            f"找不到股票代號 {stock_id}。"
        )

    today = pd.Timestamp(
        now_taipei().date()
    )

    # ========================================================
    # 上櫃：先用 TPEx OpenAPI
    # ========================================================
    if market == "TWO":
        latest = get_tpex_latest_openapi(stock_id)

        if latest:
            trade_date = latest["日期"]
            high = latest["最高價"]
            low = latest["最低價"]
            turnover = latest["成交金額"]

            # OpenAPI 在休市日會回最近交易日，因此日期照原始資料顯示
            if (
                pd.notna(high)
                and pd.notna(low)
                and pd.notna(turnover)
            ):
                price_range = high - low
                turnover_billion = turnover / 100_000_000
                ratio = (
                    turnover_billion / price_range
                    if price_range > 0
                    else np.nan
                )

                return {
                    "名稱": f"{stock_id} {latest['股票名稱']}",
                    "市場名稱": "上櫃",
                    "日期": trade_date if pd.notna(trade_date) else today,
                    "時間": "",
                    "最高價": high,
                    "最低價": low,
                    "價差": price_range,
                    "成交金額(億)": turnover_billion,
                    "成交金額價差比(億)": ratio,
                    "資料來源": "TPEx OpenAPI（最新交易日）"
                }

    # ========================================================
    # 上市 / TPEx OpenAPI 無資料時：原有官方歷史路徑
    # ========================================================
    official = get_stock_range(
        stock_id,
        market,
        today,
        today
    )

    stock_name = (
        snapshot.get("股票名稱", stock_id)
        if snapshot
        else stock_id
    )

    if not official.empty:
        row = official.iloc[-1]
        source = (
            row.get("資料來源", "官方當日資料")
            if hasattr(row, "get")
            else "官方當日資料"
        )

        return {
            "名稱": f"{stock_id} {stock_name}",
            "市場名稱": MARKET_LABEL[market],
            "日期": row["日期"],
            "時間": "",
            "最高價": row["最高價"],
            "最低價": row["最低價"],
            "價差": row["價差"],
            "成交金額(億)": row["成交金額(億)"],
            "成交金額價差比(億)": row["成交金額價差比(億)"],
            "資料來源": str(source)
        }

    # ========================================================
    # 最後才使用 Yahoo 盤中資料
    # ========================================================
    intraday = get_stock_intraday(
        stock_id,
        market
    )

    if not intraday:
        # 非交易日：上櫃仍可顯示 OpenAPI 最近交易日，理論上前面已回傳
        raise ValueError(
            "今天沒有可用交易資料。"
        )

    high = intraday["最高價"]
    low = intraday["最低價"]
    turnover = intraday["成交金額"]
    price_range = high - low
    turnover_billion = turnover / 100_000_000

    ratio = (
        turnover_billion / price_range
        if price_range > 0
        else np.nan
    )

    return {
        "名稱": f"{stock_id} {stock_name}",
        "市場名稱": MARKET_LABEL[market],
        "日期": today,
        "時間": (
            snapshot.get("時間", "")
            if snapshot
            else ""
        ),
        "最高價": high,
        "最低價": low,
        "價差": price_range,
        "成交金額(億)": turnover_billion,
        "成交金額價差比(億)": ratio,
        "資料來源": "Yahoo 1分鐘盤中估算"
    }


# ============================================================
# 19. 大盤今日
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def get_today_index():
    today = pd.Timestamp(
        now_taipei().date()
    )

    official = get_index_range(
        today,
        today
    )

    if not official.empty:
        row = official.iloc[-1]

        return {
            "名稱":
                "臺灣加權股價指數",

            "市場名稱":
                "大盤",

            "日期":
                row["日期"],

            "時間":
                "",

            "最高價":
                row["最高價"],

            "最低價":
                row["最低價"],

            "價差":
                row["價差"],

            "成交金額(億)":
                row["成交金額(億)"],

            "成交金額價差比(億)":
                row[
                    "成交金額價差比(億)"
                ],

            "資料來源":
                "TWSE 官方大盤資料"
        }

    intraday = get_taiex_intraday()

    if not intraday:
        raise ValueError(
            "目前沒有可用的大盤資料。"
        )

    month_turnover = get_market_turnover_month(
        today.year,
        today.month
    )

    turnover = np.nan

    if not month_turnover.empty:
        row = month_turnover[
            month_turnover["日期"]
            ==
            today
        ]

        if not row.empty:
            turnover = (
                row.iloc[-1][
                    "成交金額"
                ]
            )

    high = intraday["最高價"]
    low = intraday["最低價"]

    price_range = high - low

    if pd.notna(turnover):
        turnover_billion = (
            turnover
            /
            100_000_000
        )

        ratio = (
            turnover_billion
            /
            price_range
            if price_range > 0
            else np.nan
        )
    else:
        turnover_billion = np.nan
        ratio = np.nan

    return {
        "名稱":
            "臺灣加權股價指數",

        "市場名稱":
            "大盤",

        "日期":
            today,

        "時間":
            now_taipei().strftime(
                "%H:%M:%S"
            ),

        "最高價":
            high,

        "最低價":
            low,

        "價差":
            price_range,

        "成交金額(億)":
            turnover_billion,

        "成交金額價差比(億)":
            ratio,

        "資料來源":
            (
                "盤中指數 + TWSE 成交金額"
                if pd.notna(turnover)
                else
                "盤中指數；官方成交金額尚未取得"
            )
    }


# ============================================================
# 20. 單日 UI
# ============================================================

def show_single_day(data):
    st.subheader(
        f"{data['名稱']}｜"
        f"{data['市場名稱']}"
    )

    date_text = (
        pd.Timestamp(
            data["日期"]
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    if data.get("時間"):
        st.caption(
            f"資料時間："
            f"{date_text} "
            f"{data['時間']}"
        )
    else:
        st.caption(
            f"資料日期：{date_text}"
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "最高價 / 最高指數",
        format_price(
            data["最高價"]
        )
    )

    c2.metric(
        "最低價 / 最低指數",
        format_price(
            data["最低價"]
        )
    )

    c3.metric(
        "成交金額",
        format_billion(
            data["成交金額(億)"]
        )
    )

    c4.metric(
        "成交金額 / (最高 - 最低)",
        format_billion(
            data[
                "成交金額價差比(億)"
            ]
        )
    )

    st.divider()

    st.write(
        "**價差：** "
        f"{data['最高價']:.2f}"
        " - "
        f"{data['最低價']:.2f}"
        " = "
        f"**{data['價差']:.2f}**"
    )

    if pd.notna(
        data[
            "成交金額價差比(億)"
        ]
    ):
        st.write(
            "**計算：** "
            f"{data['成交金額(億)']:.2f} 億"
            " ÷ "
            f"{data['價差']:.2f}"
            " = "
            f"**{data['成交金額價差比(億)']:.2f} 億**"
        )
    else:
        st.warning(
            "目前尚未取得當日官方成交金額，"
            "因此暫時無法計算成交金額價差比。"
        )

    st.info(
        f"資料來源："
        f"{data['資料來源']}"
    )


# ============================================================
# 21. 區間 UI
# ============================================================

def show_range(
    df,
    title,
    start_date,
    end_date
):
    st.subheader(title)

    st.caption(
        f"查詢區間："
        f"{start_date} ～ {end_date}"
    )

    ratios = (
        df[
            "成交金額價差比(億)"
        ]
        .dropna()
    )

    average_ratio = (
        ratios.mean()
        if not ratios.empty
        else np.nan
    )

    max_ratio = (
        ratios.max()
        if not ratios.empty
        else np.nan
    )

    min_ratio = (
        ratios.min()
        if not ratios.empty
        else np.nan
    )

    alert_threshold = (
        average_ratio * 3
        if pd.notna(
            average_ratio
        )
        else np.nan
    )

    df = df.copy()

    df["ALERT"] = ""

    if pd.notna(alert_threshold):
        df.loc[
            (
                df[
                    "成交金額價差比(億)"
                ]
                >=
                alert_threshold
            ),
            "ALERT"
        ] = "🚨 ALERT"

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "區間平均",
        format_billion(
            average_ratio
        )
    )

    c2.metric(
        "3倍 ALERT 門檻",
        format_billion(
            alert_threshold
        )
    )

    c3.metric(
        "最大值",
        format_billion(
            max_ratio
        )
    )

    c4.metric(
        "最小值",
        format_billion(
            min_ratio
        )
    )

    c5.metric(
        "交易天數",
        f"{len(df)} 天"
    )

    alert_df = df[
        df["ALERT"]
        ==
        "🚨 ALERT"
    ]

    if not alert_df.empty:
        st.error(
            f"🚨 發現 {len(alert_df)} 個交易日"
            "達到或超過區間平均的 3 倍"
        )

        for _, row in alert_df.iterrows():
            st.warning(
                f"{row['日期'].strftime('%Y-%m-%d')}｜"
                f"{row['成交金額價差比(億)']:.2f} 億｜"
                f"門檻 {alert_threshold:.2f} 億"
            )
    else:
        st.success(
            "此區間沒有出現 >= 平均值 3 倍的資料。"
        )

    st.divider()

    if "資料來源" in df.columns and df["資料來源"].astype(str).str.contains("Yahoo").any():
        st.warning(
            "部分上櫃歷史資料因 TPEx 官方歷史介面在雲端環境暫時無法取得，"
            "已使用 Yahoo 歷史行情備援；該備援的成交金額為估算值。"
        )

    display_df = df[[
        "日期",
        "最高價",
        "最低價",
        "價差",
        "成交金額(億)",
        "成交金額價差比(億)",
        "ALERT"
    ]].copy()

    display_df["日期"] = (
        display_df["日期"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "最高價":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

            "最低價":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

            "價差":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

            "成交金額(億)":
                st.column_config.NumberColumn(
                    format="%.2f 億"
                ),

            "成交金額價差比(億)":
                st.column_config.NumberColumn(
                    format="%.2f 億"
                )
        }
    )

    csv_data = (
        display_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    st.download_button(
        "⬇️ 下載區間 CSV",
        data=csv_data,
        file_name=(
            f"{start_date}_"
            f"{end_date}.csv"
        ),
        mime="text/csv",
        width="stretch"
    )


# ============================================================
# 22. 主畫面
# ============================================================

st.title(
    "📊 台股成交金額分析"
)

st.caption(
    "個股 + 臺灣加權股價指數｜"
    "成交金額價差比 = "
    "成交金額（億） ÷（最高－最低）"
)


# ============================================================
# 23. Sidebar
# ============================================================

with st.sidebar:
    st.header(
        "查詢設定"
    )

    target_type = st.radio(
        "查詢標的",
        [
            "個股",
            "大盤指數"
        ]
    )

    if target_type == "個股":
        stock_id = (
            st.text_input(
                "股票代號",
                value="6770",
                placeholder=
                    "例如：6770、2330、5347"
            )
            .strip()
        )
    else:
        st.info(
            "目前大盤指數："
            "臺灣加權股價指數 TAIEX"
        )

    query_mode = st.radio(
        "查詢模式",
        [
            "今日",
            "指定單日",
            "日期區間"
        ]
    )

    today_date = (
        now_taipei()
        .date()
    )

    if query_mode == "指定單日":
        target_date = st.date_input(
            "選擇日期",
            value=today_date,
            max_value=today_date
        )

    elif query_mode == "日期區間":
        default_start = (
            pd.Timestamp(today_date)
            -
            pd.Timedelta(days=30)
        ).date()

        start_date = st.date_input(
            "開始日期",
            value=default_start,
            max_value=today_date
        )

        end_date = st.date_input(
            "結束日期",
            value=today_date,
            max_value=today_date
        )

    search = st.button(
        "🔍 查詢",
        type="primary",
        width="stretch"
    )

    if st.button(
        "🔄 清除快取",
        width="stretch"
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# 24. 執行
# ============================================================

if search:
    try:
        if target_type == "個股":
            if not stock_id.isdigit():
                raise ValueError(
                    "股票代號請輸入數字。"
                )

            market, snapshot = detect_market(
                stock_id
            )

            if market is None:
                raise ValueError(
                    f"找不到股票代號 {stock_id}。"
                )

            stock_name = (
                snapshot.get(
                    "股票名稱",
                    stock_id
                )
                if snapshot
                else stock_id
            )

            if query_mode == "今日":
                with st.spinner(
                    "正在取得今日個股資料..."
                ):
                    data = get_today_stock(
                        stock_id
                    )

                show_single_day(
                    data
                )

            elif query_mode == "指定單日":
                df = get_stock_range(
                    stock_id,
                    market,
                    target_date,
                    target_date
                )

                if df.empty:
                    raise ValueError(
                        f"{target_date} 沒有交易資料。"
                    )

                row = df.iloc[-1]

                data = {
                    "名稱":
                        f"{stock_id} {stock_name}",

                    "市場名稱":
                        MARKET_LABEL[market],

                    "日期":
                        row["日期"],

                    "時間":
                        "",

                    "最高價":
                        row["最高價"],

                    "最低價":
                        row["最低價"],

                    "價差":
                        row["價差"],

                    "成交金額(億)":
                        row["成交金額(億)"],

                    "成交金額價差比(億)":
                        row[
                            "成交金額價差比(億)"
                        ],

                    "資料來源":
                        "官方歷史資料"
                }

                show_single_day(
                    data
                )

            else:
                if start_date > end_date:
                    raise ValueError(
                        "開始日期不能晚於結束日期。"
                    )

                df = get_stock_range(
                    stock_id,
                    market,
                    start_date,
                    end_date
                )

                if df.empty:
                    raise ValueError(
                        "此區間沒有交易資料。"
                    )

                show_range(
                    df,
                    (
                        f"{stock_id} "
                        f"{stock_name}｜"
                        f"{MARKET_LABEL[market]}"
                    ),
                    start_date,
                    end_date
                )

        else:
            if query_mode == "今日":
                with st.spinner(
                    "正在取得今日大盤資料..."
                ):
                    data = get_today_index()

                show_single_day(
                    data
                )

            elif query_mode == "指定單日":
                df = get_index_range(
                    target_date,
                    target_date
                )

                if df.empty:
                    raise ValueError(
                        f"{target_date} 沒有大盤交易資料。"
                    )

                row = df.iloc[-1]

                data = {
                    "名稱":
                        "臺灣加權股價指數",

                    "市場名稱":
                        "大盤",

                    "日期":
                        row["日期"],

                    "時間":
                        "",

                    "最高價":
                        row["最高價"],

                    "最低價":
                        row["最低價"],

                    "價差":
                        row["價差"],

                    "成交金額(億)":
                        row["成交金額(億)"],

                    "成交金額價差比(億)":
                        row[
                            "成交金額價差比(億)"
                        ],

                    "資料來源":
                        "TWSE 官方大盤歷史資料"
                }

                show_single_day(
                    data
                )

            else:
                if start_date > end_date:
                    raise ValueError(
                        "開始日期不能晚於結束日期。"
                    )

                df = get_index_range(
                    start_date,
                    end_date
                )

                if df.empty:
                    raise ValueError(
                        "此區間沒有大盤資料。"
                    )

                show_range(
                    df,
                    "臺灣加權股價指數｜大盤",
                    start_date,
                    end_date
                )

        st.divider()

        st.caption(
            "程式更新時間："
            f"{now_taipei().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except Exception as e:
        st.error(
            str(e)
        )

        with st.expander(
            "錯誤詳細資訊"
        ):
            st.exception(
                e
            )

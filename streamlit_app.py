from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA Fuji V7 Pro BEI Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 BBMA Fuji V7 - Ultimate BEI Re-Entry Screener")
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem penyaringan fokus murni: **Validasi Sinyal & Re-Entry BBMA Fuji V7**"
    " untuk seluruh emiten Bursa Efek Indonesia."
)


# -----------------------------------------------------------------
# 1. LOAD OTOMATIS SELURUH SAHAM DARI EXCEL
# -----------------------------------------------------------------
@st.cache_data
def load_all_tickers_from_excel():
  try:
    excel_file = "Daftar Saham  - 20261111.xlsx"
    df_excel = pd.read_excel(excel_file)
    tickers = df_excel["Kode"].dropna().astype(str).str.strip().tolist()
    return sorted(list(set(tickers)))
  except Exception as e:
    st.warning(
        f"File Excel tidak ditemukan ({e}). Menggunakan cadangan emiten utama."
    )
    return ["BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "GOTO", "ADRO"]


all_stocks = load_all_tickers_from_excel()

# -----------------------------------------------------------------
# 2. SIDEBAR PENGATURAN MODE & TIMEFRAME
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Sistem Pro")
mode = st.sidebar.radio(
    "Pilih Mode Operasi",
    [
        "🚀 Auto Screener (Fuji V7 Re-Entry Valid)",
        "🔍 Cek Manual (Single Stock Pro)",
    ],
)

tf_package = st.sidebar.selectbox(
    "Pilih Paket Hierarki Timeframe BBMA",
    ["D1 - H4 - H1", "H4 - H1 - M15", "M15 - M5 - M1"],
    index=0,
)

if tf_package == "D1 - H4 - H1":
  tf_high, tf_mid, tf_exec = "1d", "4h", "1h"
elif tf_package == "H4 - H1 - M15":
  tf_high, tf_mid, tf_exec = "4h", "1h", "15m"
else:
  tf_high, tf_mid, tf_exec = "15m", "5m", "1m"


# -----------------------------------------------------------------
# 3. FUNGSI TEKNIKAL & ANALISIS BBMA FUJI V7
# -----------------------------------------------------------------
def calculate_wma(series, length):
  weights = np.arange(1, length + 1)
  return series.rolling(length).apply(
      lambda x: np.dot(x, weights) / weights.sum(), raw=True
  )


@st.cache_data(ttl=0)
def load_stock_data(ticker_symbol, interval):
  period = "5d" if interval in ["1m", "5m", "15m"] else "1mo"
  if interval == "4h":
    period = "3mo"
  elif interval == "1d":
    period = "1y"

  try:
    df = yf.download(
        ticker_symbol, period=period, interval=interval, progress=False
    )
    if not df.empty:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
      if not df.empty:
        return calculate_bbma_fuji(df)
  except Exception:
    pass
  return pd.DataFrame()


def calculate_bbma_fuji(df):
  if df.empty or len(df) < 50:
    return df
  df = df.copy()

  # Bollinger Bands (Length 20, Mult 2)
  BBlength = 20
  BBdev = 2.0
  df["midBB"] = df["Close"].rolling(window=BBlength).mean()
  stdev = df["Close"].rolling(window=BBlength).std()
  df["topBB"] = df["midBB"] + (BBdev * stdev)
  df["lowBB"] = df["midBB"] - (BBdev * stdev)

  # WMA sesuai script Fuji V7
  df["mahi5"] = calculate_wma(df["High"], 5)
  df["mahi10"] = calculate_wma(df["High"], 10)
  df["malo5"] = calculate_wma(df["Low"], 5)
  df["malo10"] = calculate_wma(df["Low"], 10)
  df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

  df["mahi5_p"] = df["mahi5"].shift(1)
  df["malo5_p"] = df["malo5"].shift(1)

  # Sinyal Fuji V7
  df["ext_buy"] = (df["Close"] > df["lowBB"]) & (
      (df["malo5_p"] < df["lowBB"]) | (df["malo5"] < df["lowBB"])
  )
  df["mmt_buy"] = (df["Close"] > df["topBB"]) & (df["Open"] < df["topBB"])
  df["ali_buy"] = (df["Open"] > df["topBB"]) & (df["Close"] > df["topBB"])

  extreme_prev_buy = df["malo5"].shift(1) < df["lowBB"].shift(1)
  is_csm_buy = df["Close"] > df["topBB"]
  df["mhv_buy"] = (
      extreme_prev_buy
      & (~is_csm_buy)
      & (df["Low"] > df["lowBB"])
      & (df["Close"] < df["midBB"])
  )

  df["csak_buy"] = (
      (df["Close"] > df["midBB"])
      & (df["Close"] > df["mahi5"])
      & (df["Close"] > df["mahi10"])
      & (df["Close"] > df["malo5"])
      & (df["Close"] > df["malo10"])
  )

  df["reject_malo"] = (
      (df["Low"] < df["malo5"])
      & (df["Close"] > df["malo5"])
      & (df["Close"] > df["malo10"])
      & (df["Close"] > df["midBB"])
      & (df["malo5"] > df["midBB"])
  )

  return df


def resample_to_h4(df_h1):
  if df_h1.empty:
    return pd.DataFrame()
  df_h4 = df_h1.resample("4h").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
      "Volume": "sum",
  })
  return calculate_bbma_fuji(df_h4.dropna())


def analyze_tf_fuji(df, role_type="TF1"):
  if df.empty or len(df) < 50:
    return {
        "price": 0.0,
        "status": "DATA KURANG",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ema50": 0.0,
        "detail": "Data tidak mencukupi",
    }

  latest = df.iloc[-1]
  c_price = float(latest["Close"])
  top_bb = float(latest["topBB"])
  bot_bb = float(latest["lowBB"])
  mid_bb = float(latest["midBB"])
  ema50 = float(latest["ema50"])

  is_ext_buy = bool(latest["ext_buy"])
  is_mhv_buy = bool(latest["mhv_buy"])
  is_csak_buy = bool(latest["csak_buy"])
  is_mmt_buy = bool(latest["mmt_buy"] or latest["ali_buy"])
  is_re_buy = bool(latest["reject_malo"])

  status = "-"
  bias = "NEUTRAL"
  detail = "-"

  if role_type == "TF1":
    if is_re_buy or (
        c_price >= bot_bb and c_price <= latest["malo10"] * 1.025
    ):
      status = "RE-ENTRY BUY VALID (Acuan Utama)"
      bias = "BUY"
      detail = "Penolakan valid di area MA Low (Fuji V7)."
    elif is_mhv_buy or is_ext_buy:
      status = "EXTREME / MHV BUY (Acuan Utama)"
      bias = "BUY"
      detail = "Terdeteksi Extreme/MHV Buy di Low BB."
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menunggu Re-Entry)"
      bias = "BUY"
      detail = "Harga di atas Mid BB, menunggu pullback."
    else:
      status = "BEARISH / TIDAK VALID"
      bias = "SELL"
      detail = "Di bawah Mid BB."

  elif role_type == "TF2":
    if is_re_buy or is_ext_buy or is_mhv_buy or is_mmt_buy or is_csak_buy:
      status = "KONFIRMASI BUY VALID"
      bias = "BUY"
      detail = "TF2 mengonfirmasi struktur arah Buy."
    elif c_price > mid_bb:
      status = "TF2 BULLISH"
      bias = "BUY"
      detail = "Di atas Mid BB."
    else:
      status = "TF2 BELUM KONFIRMASI"
      bias = "SELL"
      detail = "Belum ada konfirmasi."

  elif role_type == "TF3":
    if is_mmt_buy or is_csak_buy or is_re_buy:
      status = "TRIGGER EKSEKUSI BUY"
      bias = "BUY"
      detail = "TF3 memicu Momentum / CSAK / Re-Entry."
    elif c_price > mid_bb:
      status = "TF3 BULLISH"
      bias = "BUY"
      detail = "Mengalir searah."
    else:
      status = "TF3 BELUM ADA TRIGGER"
      bias = "SELL"
      detail = "Belum ada trigger."

  return {
      "price": c_price,
      "status": status,
      "bias": bias,
      "top_bb": top_bb,
      "bot_bb": bot_bb,
      "mid_bb": mid_bb,
      "ema50": ema50,
      "detail": detail,
  }


def evaluate_pure_bbma_screener(code, tf_high_code, tf_mid_code, tf_exec_code):
  try:
    d_exec = load_stock_data(f"{code}.JK", tf_exec_code)
    if d_exec.empty or len(d_exec) < 25:
      return None

    latest = d_exec.iloc[-1]
    val_trx = latest["Close"] * latest["Volume"]

    # Evaluasi Struktur Multi-Timeframe BBMA Fuji V7
    if tf_high_code == "4h":
      df_h1 = load_stock_data(f"{code}.JK", "1h")
      d_high = resample_to_h4(df_h1)
    else:
      d_high = load_stock_data(f"{code}.JK", tf_high_code)

    d_mid = load_stock_data(f"{code}.JK", tf_mid_code)
    if d_high.empty or d_mid.empty:
      return None

    res_tf1 = analyze_tf_fuji(d_high, role_type="TF1")
    res_tf2 = analyze_tf_fuji(d_mid, role_type="TF2")
    res_tf3 = analyze_tf_fuji(d_exec, role_type="TF3")

    is_bbma_valid = (
        res_tf1["bias"] == "BUY"
        and res_tf2["bias"] == "BUY"
        and res_tf3["bias"] == "BUY"
    )

    if not is_bbma_valid:
      return None

    category = "🎯 RE-ENTRY / SETUP BBMA VALID"
    entry_zone = (res_tf3["bot_bb"] + res_tf3["mid_bb"]) / 2

    return {
        "Emiten": code,
        "Harga Terkini": res_tf3["price"],
        "Nilai Transaksi": f"Rp {val_trx:,.0f}",
        "Estimasi Entry": f"Rp {entry_zone:,.0f}",
        "Kategori Sinyal": category,
        "DF_Exec": d_exec,
        "TF1": res_tf1,
        "TF2": res_tf2,
        "TF3": res_tf3,
    }
  except Exception:
    return None


# -----------------------------------------------------------------
# 4. TAMPILAN DASHBOARD UTAMA (STREAMLIT UI)
# -----------------------------------------------------------------
if mode == "🚀 Auto Screener (Fuji V7 Re-Entry Valid)":
  st.subheader(
      "🚀 Auto Screener - Murni Validasi Struktur BBMA Fuji V7 (Multi-Timeframe)"
  )
  st.markdown(
      "Sistem memindai seluruh emiten BEI untuk mencari saham yang memenuhi"
      " aturan valid re-entry / setup Buy Fuji V7 di semua timeframe pilihan"
      " Anda."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 60
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING BBMA")

  if run_scan:
    stocks_to_scan = all_stocks[:scan_limit]
    prog_bar = st.progress(0)
    txt_status = st.empty()

    results = []
    total = len(stocks_to_scan)

    for i, code in enumerate(stocks_to_scan):
      txt_status.text(
          f"Memindai & Menganalisis [{i+1}/{total}]: {code}.JK ..."
      )
      res = evaluate_pure_bbma_screener(code, tf_high, tf_mid, tf_exec)
      if res:
        results.append(res)
      prog_bar.progress((i + 1) / total)

    txt_status.empty()
    prog_bar.empty()

    if results:
      df_res = pd.DataFrame(results)

      st.markdown("---")
      st.subheader("🔥 HASIL PEMINDAIAN BBMA VALID")
      st.success(
          f"Ditemukan **{len(df_res)} emiten** dengan struktur BBMA Valid!"
      )

      st.dataframe(
          df_res[
              [
                  "Emiten",
                  "Harga Terkini",
                  "Nilai Transaksi",
                  "Estimasi Entry",
                  "Kategori Sinyal",
              ]
          ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
          use_container_width=True,
      )

      selected_valid = st.selectbox(
          "Pilih emiten di atas untuk melihat detail chart & struktur BBMA:",
          df_res["Emiten"].tolist(),
      )
      matched_row = df_res[df_res["Emiten"] == selected_valid].iloc[0]
      d_target = matched_row["DF_Exec"]

      fig = go.Figure()
      fig.add_trace(
          go.Candlestick(
              x=d_target.index,
              open=d_target["Open"],
              high=d_target["High"],
              low=d_target["Low"],
              close=d_target["Close"],
              name="Harga",
          )
      )
      fig.add_trace(
          go.Scatter(
              x=d_target.index,
              y=d_target["topBB"],
              name="BB Upper",
              line=dict(color="gray", width=1, dash="dot"),
          )
      )
      fig.add_trace(
          go.Scatter(
              x=d_target.index,
              y=d_target["midBB"],
              name="Mid BB",
              line=dict(color="orange", width=1.5),
          )
      )
      fig.add_trace(
          go.Scatter(
              x=d_target.index,
              y=d_target["lowBB"],
              name="BB Lower",
              line=dict(color="gray", width=1, dash="dot"),
          )
      )
      fig.add_trace(
          go.Scatter(
              x=d_target.index,
              y=d_target["ema50"],
              name="EMA 50",
              line=dict(color="magenta", width=1.2),
          )
      )

      fig.update_layout(
          title=(
              f"Chart Analisis BBMA - {selected_valid}.JK ({tf_exec.upper()})"
          ),
          yaxis_title="Harga (IDR)",
          template="plotly_dark",
          height=500,
          xaxis_rangeslider_visible=False,
      )
      st.plotly_chart(fig, use_container_width=True)

    else:
      st.warning(
          "⚠️ Tidak ada emiten yang memenuhi struktur valid pada pemindaian"
          " ini."
      )
  else:
    st.info(
        "👈 Pilih paket timeframe di sidebar, lalu klik **⚡ MULAI SCANNING"
        " BBMA** untuk memindai pasar BEI."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL
# -----------------------------------------------------------------
else:
  st.subheader("🔍 Cek Manual Pro - Analisis Mendalam Emiten (BBMA Fuji V7)")
  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(
      f"Menganalisis data mendalam untuk {selected_stock}.JK ..."
  ):
    res = evaluate_pure_bbma_screener(
        selected_stock, tf_high, tf_mid, tf_exec
    )
    d_exec = load_stock_data(f"{selected_stock}.JK", tf_exec)

  if d_exec.empty:
    st.warning(
        f"⚠️ Gagal memuat data untuk **{selected_stock}.JK**. Periksa likuiditas"
        " saham tersebut."
    )
  else:
    latest_price = d_exec["Close"].iloc[-1]
    prev_price = d_exec["Close"].iloc[-2] if len(d_exec) > 1 else latest_price
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100 if prev_price > 0 else 0

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
    c1.metric(
        f"Harga {selected_stock}",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric("Volume Hari Ini", f"{d_exec['Volume'].iloc[-1]:,.0f}")
    c3.metric(
        "Nilai Transaksi", f"Rp {latest_price * d_exec['Volume'].iloc[-1]:,.0f}"
    )

    # Status Kombo / Setup BBMA
    status_text = (
        res["Kategori Sinyal"]
        if res
        else "Belum Valid (Struktur Belum Memenuhi)"
    )
    c4.markdown(
        f"""
        <div style="background-color: #262730; padding: 10px 14px; border-radius: 4px; border: 1px solid #41424C;">
            <p style="color: #9N9N9N; font-size: 12px; margin-bottom: 2px;">Status Setup BBMA</p>
            <p style="color: #FFFFFF; font-size: 15px; font-weight: 600; margin: 0; word-break: break-word;">{status_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if res:
      st.success(
          f"✨ **{selected_stock}** memiliki struktur & Re-Entry BBMA Fuji V7"
          " yang **VALID**!"
      )
    else:
      st.info(
          f"ℹ️ **{selected_stock}** belum memenuhi syarat validasi multi-timeframe"
          " BBMA saat ini."
      )

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d_exec.index,
            open=d_exec["Open"],
            high=d_exec["High"],
            low=d_exec["Low"],
            close=d_exec["Close"],
            name="Harga",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["topBB"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["midBB"],
            name="Mid BB",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["lowBB"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["ema50"],
            name="EMA 50",
            line=dict(color="magenta", width=1.2),
        )
    )

    fig.update_layout(
        title=f"Grafik Analisis BBMA - {selected_stock}.JK",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

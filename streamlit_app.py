from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA MTF Full BEI Screener", page_icon="📈", layout="wide"
)

st.title("📈 BBMA Oma Ally Screener & Batch Filter (Full 944+ Saham BEI)")
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem otomatis memindai seluruh emiten Bursa Efek Indonesia (BEI) dari database file."
)


# -----------------------------------------------------------------
# LOAD SEMUA KODE SAHAM DARI FILE EXCEL DI GITHUB
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
        f"File Excel tidak ditemukan ({e}). Menggunakan daftar cadangan standar."
    )
    return ["BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "GOTO", "ADRO"]


all_stocks = load_all_tickers_from_excel()

# Sidebar Pengaturan
st.sidebar.header("⚙️ Konfigurasi Mode Analisis")
mode = st.sidebar.radio(
    "Pilih Mode", ["🔍 Analisis Mandiri (Single Stock)", "🚀 Batch Scan & Filter Semua Saham"]
)

if mode == "🔍 Analisis Mandiri (Single Stock)":
  selected_stock = st.sidebar.selectbox(
      f"Pilih dari Total {len(all_stocks)} Saham BEI", all_stocks, index=0
  )
  ticker = f"{selected_stock}.JK"

  tf_exec = st.sidebar.selectbox(
      "Timeframe Eksekusi (TF 1)", ["15m", "30m", "1h"], index=0
  )
  tf_mid = st.sidebar.selectbox(
      "Timeframe Validasi (TF 2)", ["1h", "4h"], index=0
  )
  tf_high = st.sidebar.selectbox(
      "Timeframe Tren Utama (TF 3)", ["1d"], index=0
  )


  @st.cache_data(ttl=0)
  def load_stock_data(ticker_symbol, interval, period="1mo"):
    try:
      df = yf.download(
          ticker_symbol, period=period, interval=interval, progress=False
      )
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df.dropna(inplace=True)
      return df
    except Exception:
      return pd.DataFrame()


  def calculate_bbma(df):
    if df.empty or len(df) < 20:
      return df
    df["MA5_High"] = df["High"].rolling(window=5).mean()
    df["MA5_Low"] = df["Low"].rolling(window=5).mean()
    df["MA10_High"] = df["High"].rolling(window=10).mean()
    df["MA10_Low"] = df["Low"].rolling(window=10).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (std * 2)
    return df


  data_exec = calculate_bbma(load_stock_data(ticker, tf_exec, "5d"))
  data_mid = calculate_bbma(load_stock_data(ticker, tf_mid, "1mo"))
  data_high = calculate_bbma(load_stock_data(ticker, tf_high, "3mo"))

  if data_exec.empty or data_mid.empty:
    st.warning(
        f"⚠️ Data untuk **{selected_stock}.JK** tidak ditemukan / likuiditas rendah."
    )
  else:
    latest_price = data_exec["Close"].iloc[-1]
    prev_price = data_exec["Close"].iloc[-2]
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Emiten ({selected_stock})",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric("TF 1 (Eksekusi)", tf_exec.upper())
    c3.metric("TF 2 (Validasi)", tf_mid.upper())
    c4.metric("TF 3 (Tren Utama)", tf_high.upper())

    st.markdown("---")


    def check_mtf_validation(d_exec, d_mid, d_high):
      high_bullish = (
          d_high["Close"].iloc[-1] > d_high["EMA50"].iloc[-1]
          if not d_high.empty
          else True
      )
      mid_bullish = (
          d_mid["Close"].iloc[-1] > d_mid["BB_Middle"].iloc[-1]
          if not d_mid.empty
          else True
      )

      exec_row = d_exec.iloc[-1]
      is_near_ma_low = (
          exec_row["Low"] <= exec_row["MA10_Low"] * 1.02
          and exec_row["Close"] > exec_row["MA50"]
          if "MA50" in d_exec.columns
          else True
      )

      re_entry_buy = is_near_ma_low and mid_bullish and high_bullish
      return high_bullish, mid_bullish, re_entry_buy


    h_bull, m_bull, is_reentry = check_mtf_validation(
        data_exec, data_mid, data_high
    )

    st.subheader("🎯 Hasil Analisis Validasi Multi-Timeframe (MTF)")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        f"TF 3 ({tf_high.upper()}) Trend",
        "Bullish (Aman)" if h_bull else "Bearish / Konsolidasi",
    )
    col_b.metric(
        f"TF 2 ({tf_mid.upper()}) Validasi",
        "Valid Buy Support" if m_bull else "Belum Valid",
    )
    col_c.metric(
        f"TF 1 ({tf_exec.upper()}) Signal",
        "RE-ENTRY BUY VALID 🟢" if is_reentry else "Tunggu Setup / Flat ⏳",
    )

    if is_reentry:
      st.success(
          f"**SINYAL VALID RE-ENTRY BUY UNTUK SAHAM {selected_stock}**\n\n"
          "- **TF 3 & TF 2:** Tren mendukung kenaikan.\n"
          "- **TF 1 (Eksekusi):** Harga memantul di area MA Low (Zona Re-Entry Buy sah)."
      )
    else:
      st.info(
          f"Belum memenuhi syarat mutlak Re-Entry Buy Multi-Timeframe untuk **{selected_stock}**."
      )

    st.markdown("---")

    st.subheader(
        f"📊 Chart BBMA TF 1 ({tf_exec.upper()}) - {selected_stock}.JK"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data_exec.index,
            open=data_exec["Open"],
            high=data_exec["High"],
            low=data_exec["Low"],
            close=data_exec["Close"],
            name="Harga Saham",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_exec.index,
            y=data_exec["BB_Upper"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_exec.index,
            y=data_exec["BB_Middle"],
            name="BB Mid",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_exec.index,
            y=data_exec["BB_Lower"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_exec.index,
            y=data_exec["MA5_Low"],
            name="MA5 Low",
            line=dict(color="blue", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_exec.index,
            y=data_exec["MA10_Low"],
            name="MA10 Low",
            line=dict(color="red", width=1),
        )
    )

    fig.update_layout(
        title=f"Analisis BBMA Saham {selected_stock}",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
  st.subheader("🚀 Batch Screener & Auto-Filter Seluruh Saham BEI")
  st.markdown(
      f"Sistem akan memindai emiten dari file database Anda untuk memfilter mana saja yang sedang valid di titik **Re-Entry Buy**."
  )

  scan_tf = st.sidebar.selectbox(
      "Pilih Timeframe Scan", ["1h", "4h", "1d"], index=0
  )
  scan_limit = st.sidebar.slider(
      "Batasi Jumlah Scan (Agar Proses Cepat)", 10, len(all_stocks), 50
  )
  run_button = st.sidebar.button("⚡ Mulai Scan Pasar")


  @st.cache_data(ttl=300)
  def scan_single_stock(code, interval):
    try:
      df = yf.download(
          f"{code}.JK", period="1mo", interval=interval, progress=False
      )
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      if df.empty or len(df) < 20:
        return None

      df["MA5_Low"] = df["Low"].rolling(window=5).mean()
      df["MA10_Low"] = df["Low"].rolling(window=10).mean()
      df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
      df["BB_Middle"] = df["Close"].rolling(window=20).mean()

      last = df.iloc[-1]
      is_bullish = (last["Close"] > last["EMA50"]) and (
          last["Close"] > last["BB_Middle"]
      )
      is_reentry = (last["Low"] <= last["MA10_Low"] * 1.015) and (
          last["Close"] > last["MA5_Low"]
      )

      if is_bullish and is_reentry:
        status = "🟢 RE-ENTRY BUY VALID"
        is_valid = True
      elif is_bullish:
        status = "⏳ Menunggu Retrace"
        is_valid = False
      else:
        status = "🔴 Bearish"
        is_valid = False

      return {
          "Emiten": code,
          "Harga Terkini": last["Close"],
          "Status Sinyal": status,
          "Valid": is_valid,
      }
    except Exception:
      return None


  if run_button:
    stocks_to_scan = all_stocks[:scan_limit]
    progress_bar = st.progress(0)
    status_text = st.empty()

    scan_results = []
    total = len(stocks_to_scan)

    for i, code in enumerate(stocks_to_scan):
      status_text.text(f"Memindai emiten [{i+1}/{total}]: {code}.JK ...")
      res = scan_single_stock(code, scan_tf)
      if res:
        scan_results.append(res)
      progress_bar.progress((i + 1) / total)

    status_text.empty()
    progress_bar.empty()

    df_scan = pd.DataFrame(scan_results)

    if not df_scan.empty:
      valid_df = df_scan[df_scan["Valid"] == True]
      if not valid_df.empty:
        st.success(
            f"Ditemukan **{len(valid_df)} emiten** yang sedang berada di titik **Re-Entry Buy** yang valid!"
        )
        st.dataframe(
            valid_df[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
                {"Harga Terkini": "Rp {:,.0f}"}
            ),
            use_container_width=True,
        )
      else:
        st.warning(
            "Belum ada emiten yang menyentuh zona valid Re-Entry Buy pada batas scan ini."
        )

      st.markdown("---")
      st.subheader("📋 Seluruh Hasil Pemantauan Pasar")
      st.dataframe(
          df_scan[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
              {"Harga Terkini": "Rp {:,.0f}"}
          ),
          use_container_width=True,
      )
    else:
      st.error("Gagal memuat hasil pemindaian pasar.")
  else:
    st.info(
        "👈 Atur jumlah batasan scan di sidebar, lalu klik **Mulai Scan Pasar** untuk memindai emiten dari file Excel Anda."
    )

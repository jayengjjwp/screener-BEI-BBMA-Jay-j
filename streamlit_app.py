from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA MTF Full Screener BEI", page_icon="📈", layout="wide"
)

st.title("📈 BBMA Oma Ally Multi-Timeframe & Full Screener (944+ Saham BEI)")
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem validasi hierarki tren (TF3 ➔ TF2 ➔ TF1) khusus emiten Bursa Efek Indonesia."
)


# -----------------------------------------------------------------
# 1. LOAD DATABASE SEMUA SAHAM DARI FILE EXCEL DI GITHUB
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
st.sidebar.header("⚙️ Konfigurasi Sistem")
mode = st.sidebar.radio(
    "Pilih Mode Analisis",
    ["🔍 Analisis Detail (Single Stock)", "🚀 Batch Scan & Filter Semua Saham"],
)

# Pilihan Kombinasi Timeframe Sesuai Keinginan Anda
tf_package = st.sidebar.selectbox(
    "Pilih Paket Hirarki Timeframe",
    ["D1 - H4 - H1", "H4 - H1 - M15", "M15 - M5 - M1"],
)

# Mapping Timeframe Berdasarkan Paket
if tf_package == "D1 - H4 - H1":
  tf_high, tf_mid, tf_exec = "1d", "4h", "1h"
elif tf_package == "H4 - H1 - M15":
  tf_high, tf_mid, tf_exec = "4h", "1h", "15m"
else:  # M15 - M5 - M1
  tf_high, tf_mid, tf_exec = "15m", "5m", "1m"


# -----------------------------------------------------------------
# 3. FUNGSI UTAMA KALKULATOR BBMA & DATA
# -----------------------------------------------------------------
@st.cache_data(ttl=0)
def load_stock_data(ticker_symbol, interval):
  # Menyesuaikan periode berdasarkan interval agar data cukup untuk dihitung
  period = "5d" if interval in ["1m", "5m", "15m"] else "1mo"
  if interval in ["1d", "4h"]:
    period = "3mo"

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


# -----------------------------------------------------------------
# MODE 1: ANALISIS DETAIL SATU SAHAM (SINGLE STOCK)
# -----------------------------------------------------------------
if mode == "🔍 Analisis Detail (Single Stock)":
  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )
  ticker = f"{selected_stock}.JK"

  st.subheader(
      f"📊 Analisis Mendalam: {selected_stock}.JK ({tf_package})"
  )

  with st.spinner("Sedang menarik data Multi-Timeframe..."):
    df_exec = calculate_bbma(load_stock_data(ticker, tf_exec))
    df_mid = calculate_bbma(load_stock_data(ticker, tf_mid))
    df_high = calculate_bbma(load_stock_data(ticker, tf_high))

  if df_exec.empty or df_mid.empty or df_high.empty:
    st.warning(
        f"⚠️ Gagal memuat data lengkap untuk **{selected_stock}.JK**. Kemungkinan"
        " timeframe kecil tidak didukung penuh atau likuiditas rendah."
    )
  else:
    latest_price = df_exec["Close"].iloc[-1]
    prev_price = df_exec["Close"].iloc[-2]
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Harga {selected_stock}",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric("TF Eksekusi (TF1)", tf_exec.upper())
    c3.metric("TF Validasi (TF2)", tf_mid.upper())
    c4.metric("TF Tren Utama (TF3)", tf_high.upper())

    st.markdown("---")


    # Validasi Logika BBMA MTF
    def check_mtf(d_exec, d_mid, d_high):
      h_bull = d_high["Close"].iloc[-1] > d_high["EMA50"].iloc[-1]
      m_bull = d_mid["Close"].iloc[-1] > d_mid["BB_Middle"].iloc[-1]

      exec_row = d_exec.iloc[-1]
      is_reentry = (
          exec_row["Low"] <= exec_row["MA10_Low"] * 1.015
          and exec_row["Close"] > exec_row["MA5_Low"]
          and exec_row["Close"] > exec_row["EMA50"]
      )
      valid_buy = is_reentry and m_bull and h_bull
      return h_bull, m_bull, valid_buy


    h_ok, m_ok, is_valid_reentry = check_mtf(df_exec, df_mid, df_high)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        f"TF3 Trend ({tf_high})", "Bullish (Aman)" if h_ok else "Bearish"
    )
    col_b.metric(
        f"TF2 Validasi ({tf_mid})", "Mid BB Menengadah" if m_ok else "Belum Valid"
    )
    col_c.metric(
        f"TF1 Sinyal ({tf_exec})",
        "RE-ENTRY BUY VALID 🟢" if is_valid_reentry else "Tunggu Setup ⏳",
    )

    if is_valid_reentry:
      st.success(
          f"**SINYAL VALID!** Saham **{selected_stock}** memenuhi syarat"
          f" hierarki Re-Entry Buy dari {tf_high} ➔ {tf_mid} ➔ {tf_exec}."
      )
    else:
      st.info(
          f"Saham **{selected_stock}** belum menyentuh titik konfirmasi Re-Entry"
          " Buy pada timeframe eksekusi saat ini."
      )

    st.markdown("---")

    # Visualisasi Chart Plotly TF Eksekusi
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_exec.index,
            open=df_exec["Open"],
            high=df_exec["High"],
            low=df_exec["Low"],
            close=df_exec["Close"],
            name="Harga",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exec.index,
            y=df_exec["BB_Upper"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exec.index,
            y=df_exec["BB_Middle"],
            name="BB Mid",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exec.index,
            y=df_exec["BB_Lower"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exec.index,
            y=df_exec["MA5_Low"],
            name="MA5 Low",
            line=dict(color="blue", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exec.index,
            y=df_exec["MA10_Low"],
            name="MA10 Low",
            line=dict(color="red", width=1),
        )
    )

    fig.update_layout(
        title=f"Chart BBMA Eksekusi ({tf_exec}) - {selected_stock}.JK",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------
# MODE 2: BATCH SCREENER & AUTO-FILTER SEMUA SAHAM
# -----------------------------------------------------------------
else:
  st.subheader("🚀 Batch Screener Otomatis (Filter Re-Entry Buy)")
  st.markdown(
      f"Sistem akan memindai emiten secara otomatis menggunakan hierarki"
      f" **{tf_package}** untuk memfilter emiten yang **valid Re-Entry Buy**."
  )

  scan_limit = st.sidebar.slider(
      "Batasi Jumlah Scan (Maksimal / Bertahap)", 10, len(all_stocks), 50
  )
  run_scan_btn = st.sidebar.button("⚡ Mulai Scan Semua Saham")


  def analyze_batch_stock(code):
    try:
      d_exec = load_stock_data(code, tf_exec)
      d_mid = load_stock_data(code, tf_mid)
      d_high = load_stock_data(code, tf_high)

      if d_exec.empty or d_mid.empty or d_high.empty:
        return None

      d_exec = calculate_bbma(d_exec)
      d_mid = calculate_bbma(d_mid)
      d_high = calculate_bbma(d_high)

      h_bull = d_high["Close"].iloc[-1] > d_high["EMA50"].iloc[-1]
      m_bull = d_mid["Close"].iloc[-1] > d_mid["BB_Middle"].iloc[-1]

      exec_row = d_exec.iloc[-1]
      is_reentry = (
          exec_row["Low"] <= exec_row["MA10_Low"] * 1.015
          and exec_row["Close"] > exec_row["MA5_Low"]
          and exec_row["Close"] > exec_row["EMA50"]
      )

      valid_buy = is_reentry and m_bull and h_bull

      if valid_buy:
        status = "🟢 RE-ENTRY BUY VALID"
      elif h_bull and m_bull:
        status = "⏳ Menunggu Retrace ke MA Low"
      else:
        status = "🔴 Belum Memenuhi Syarat Tren"

      return {
          "Emiten": code,
          "Harga Terkini": exec_row["Close"],
          "Status Sinyal": status,
          "Valid": valid_buy,
      }
    except Exception:
      return None


  if run_scan_btn:
    stocks_to_scan = all_stocks[:scan_limit]
    prog_bar = st.progress(0)
    txt_status = st.empty()

    scan_outputs = []
    total_scan = len(stocks_to_scan)

    for idx, c in enumerate(stocks_to_scan):
      txt_status.text(
          f"Memindai [{idx+1}/{total_scan}] menggunakan {tf_package}:"
          f" {c}.JK ..."
      )
      res = analyze_batch_stock(c)
      if res:
        scan_outputs.append(res)
      prog_bar.progress((idx + 1) / total_scan)

    txt_status.empty()
    prog_bar.empty()

    df_result = pd.DataFrame(scan_outputs)

    if not df_result.empty:
      df_valid = df_result[df_result["Valid"] == True]

      if not df_valid.empty:
        st.success(
            f"Ditemukan **{len(df_valid)} emiten** yang sedang berada di titik"
            " **Re-Entry Buy** yang valid!"
        )
        st.dataframe(
            df_valid[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
                {"Harga Terkini": "Rp {:,.0f}"}
            ),
            use_container_width=True,
        )
      else:
        st.warning(
            "Belum ada emiten yang menyentuh zona valid Re-Entry Buy pada"
            f" batasan scan dan timeframe **{tf_package}** saat ini."
        )

      st.markdown("---")
      st.subheader("📋 Tabel Hasil Pemantauan Keseluruhan")
      st.dataframe(
          df_result[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
              {"Harga Terkini": "Rp {:,.0f}"}
          ),
          use_container_width=True,
      )
    else:
      st.error("Gagal memproses pemindaian pasar.")
  else:
    st.info(
        "👈 Pilih batasan jumlah scan di sidebar, lalu klik **Mulai Scan Semua"
        " Saham** untuk memfilter emiten secara otomatis."
    )

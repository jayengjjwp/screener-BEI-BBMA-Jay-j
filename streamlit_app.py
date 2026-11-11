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

st.title("📈 BBMA Oma Ally Multi-Timeframe Screener (Full 944+ Saham BEI)")
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem validasi hierarki tren (TF3 ➔ TF2 ➔ TF1) khusus emiten Bursa Efek Indonesia."
)


# -----------------------------------------------------------------
# 1. LOAD OTOMATIS SELURUH SAHAM DARI EXCEL DI GITHUB
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
    "Pilih Mode Operasi",
    ["🚀 Auto Screener (Cari Sinyal Re-Entry)", "🔍 Cek Manual (Single Stock)"],
)

tf_package = st.sidebar.selectbox(
    "Pilih Paket Hierarki Timeframe",
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
# 3. FUNGSI TEKNIKAL & DATA
# -----------------------------------------------------------------
@st.cache_data(ttl=0)
def load_stock_data(ticker_symbol, interval):
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


def analyze_stock_logic(code):
  try:
    d_exec = calculate_bbma(load_stock_data(f"{code}.JK", tf_exec))
    d_mid = calculate_bbma(load_stock_data(f"{code}.JK", tf_mid))
    d_high = calculate_bbma(load_stock_data(f"{code}.JK", tf_high))

    if d_exec.empty or d_mid.empty or d_high.empty:
      return None

    h_bull = d_high["Close"].iloc[-1] > d_high["EMA50"].iloc[-1]
    m_bull = d_mid["Close"].iloc[-1] > d_mid["BB_Middle"].iloc[-1]

    exec_row = d_exec.iloc[-1]
    is_reentry = (
        exec_row["Low"] <= exec_row["MA10_Low"] * 1.015
        and exec_row["Close"] > exec_row["MA5_Low"]
        and exec_row["Close"] > exec_row["EMA50"]
    )

    valid_buy = is_reentry and m_bull and h_bull

    # Perkiraan Titik Entry (Kisaran MA Low)
    ma_low_zone = (exec_row["MA5_Low"] + exec_row["MA10_Low"]) / 2

    if valid_buy:
      status = "🟢 RE-ENTRY BUY VALID"
    elif h_bull and m_bull:
      status = "⏳ Menunggu Retrace"
    else:
      status = "🔴 Belum Valid"

    return {
        "Emiten": code,
        "Harga Terkini": exec_row["Close"],
        "Estimasi Area Entry (MA Low)": f"Rp {ma_low_zone:,.0f}",
        "Status Sinyal": status,
        "Valid": valid_buy,
        "DF_Exec": d_exec,
        "DF_Mid": d_mid,
        "DF_High": d_high,
    }
  except Exception:
    return None


# -----------------------------------------------------------------
# MODE 1: AUTO SCREENER (PENCARIAN OTOMATIS)
# -----------------------------------------------------------------
if mode == "🚀 Auto Screener (Cari Sinyal Re-Entry)":
  st.subheader("🚀 Auto Screener & Filter Re-Entry Buy")
  st.markdown(
      "Sistem memindai emiten secara otomatis untuk menyaring mana saja yang"
      f" **paling layak** berada di posisi **Re-Entry Buy** menggunakan"
      f" hierarki **{tf_package}**."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 80
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING PASAR")

  if run_scan:
    stocks_to_scan = all_stocks[:scan_limit]
    prog_bar = st.progress(0)
    txt_status = st.empty()

    results = []
    total = len(stocks_to_scan)

    for i, code in enumerate(stocks_to_scan):
      txt_status.text(f"Memindai emiten [{i+1}/{total}]: {code}.JK ...")
      res = analyze_stock_logic(code)
      if res:
        results.append(res)
      prog_bar.progress((i + 1) / total)

    txt_status.empty()
    prog_bar.empty()

    if results:
      df_res = pd.DataFrame(results)
      df_valid = df_res[df_res["Valid"] == True]

      st.markdown("---")
      st.subheader("🔥 HASIL FILTER: SAHAM VALID RE-ENTRY BUY")

      if not df_valid.empty:
        st.success(
            f"Ditemukan **{len(df_valid)} emiten** yang paling layak Re-Entry"
            " Buy!"
        )
        st.dataframe(
            df_valid[
                [
                    "Emiten",
                    "Harga Terkini",
                    "Estimasi Area Entry (MA Low)",
                    "Status Sinyal",
                ]
            ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
            use_container_width=True,
        )

        # Interaktif Chart untuk Emiten yang Valid
        selected_valid = st.selectbox(
            "Pilih emiten valid di atas untuk melihat chart:",
            df_valid["Emiten"].tolist(),
        )
        matched_row = df_valid[df_valid["Emiten"] == selected_valid].iloc[0]
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
                y=d_target["BB_Upper"],
                name="BB Upper",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["BB_Middle"],
                name="BB Mid",
                line=dict(color="orange", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["BB_Lower"],
                name="BB Lower",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["MA5_Low"],
                name="MA5 Low",
                line=dict(color="blue", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["MA10_Low"],
                name="MA10 Low",
                line=dict(color="red", width=1),
            )
        )

        fig.update_layout(
            title=f"Chart BBMA Eksekusi - {selected_valid}.JK",
            yaxis_title="Harga (IDR)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      else:
        st.warning(
            "Belum ada emiten yang memenuhi kriteria mutlak Re-Entry Buy saat"
            " ini."
        )

      st.markdown("---")
      st.subheader("📋 Seluruh Hasil Pemantauan Pasar")
      st.dataframe(
          df_res[
              [
                  "Emiten",
                  "Harga Terkini",
                  "Estimasi Area Entry (MA Low)",
                  "Status Sinyal",
              ]
          ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
          use_container_width=True,
      )
    else:
      st.error("Gagal menarik data pemindaian pasar.")
  else:
    st.info(
        "👈 Atur batasan scan di sidebar, lalu klik **⚡ MULAI SCANNING PASAR**"
        " untuk memfilter otomatis."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL (SINGLE STOCK ANALYSIS)
# -----------------------------------------------------------------
else:
  st.subheader("🔍 Cek Manual & Analisis Mendalam Emiten")
  st.markdown(
      "Pilih emiten apa saja secara bebas untuk melihat detail validasi"
      " Multi-Timeframe dan chart BBMA-nya."
  )

  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(f"Memuat data analisis untuk {selected_stock}.JK ..."):
    res = analyze_stock_logic(selected_stock)

  if not res or res["DF_Exec"].empty:
    st.warning(
        f"⚠️ Gagal memuat data untuk **{selected_stock}.JK**. Periksa likuiditas"
        " saham tersebut."
    )
  else:
    d_exec = res["DF_Exec"]
    latest_price = d_exec["Close"].iloc[-1]
    prev_price = d_exec["Close"].iloc[-2]
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
    st.subheader(f"🎯 Status Sinyal Manual: {res['Status Sinyal']}")
    st.markdown(
        f"📍 **Estimasi Area Titik Entry (MA Low):**"
        f" **{res['Estimasi Area Entry (MA Low)']}**"
    )

    if res["Valid"]:
      st.success(
          f"Saham **{selected_stock}** saat ini berada di posisi **Re-Entry"
          f" Buy yang Valid** menggunakan hierarki {tf_package}!"
      )
    else:
      st.info(
          f"Saham **{selected_stock}** belum memenuhi syarat mutlak Re-Entry"
          " Buy pada saat ini."
      )

    st.markdown("---")

    # Chart Detail
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
            y=d_exec["BB_Upper"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["BB_Middle"],
            name="BB Mid",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["BB_Lower"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["MA5_Low"],
            name="MA5 Low",
            line=dict(color="blue", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["MA10_Low"],
            name="MA10 Low",
            line=dict(color="red", width=1),
        )
    )

    fig.update_layout(
        title=f"Analisis Detail BBMA - {selected_stock}.JK ({tf_exec})",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

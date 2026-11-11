from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="Auto Screener BBMA MTF BEI", page_icon="🎯", layout="wide"
)

st.title(
    "🎯 Auto Screener & Filter Re-Entry Buy (Seluruh 944+ Saham BEI)"
)
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem otomatis memindai seluruh emiten dari database untuk mencari emiten yang paling layak **Re-Entry Buy**."
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
# 2. SIDEBAR PENGATURAN PILIHAN TIMEFRAME
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Pemindai Otomatis")

tf_package = st.sidebar.selectbox(
    "Pilih Hierarki Timeframe",
    ["D1 - H4 - H1", "H4 - H1 - M15", "M15 - M5 - M1"],
    index=0,
)

if tf_package == "D1 - H4 - H1":
  tf_high, tf_mid, tf_exec = "1d", "4h", "1h"
elif tf_package == "H4 - H1 - M15":
  tf_high, tf_mid, tf_exec = "4h", "1h", "15m"
else:
  tf_high, tf_mid, tf_exec = "15m", "5m", "1m"

# Batasan scan agar proses optimal dan tidak timeout
scan_limit = st.sidebar.slider(
    "Jumlah Maksimal Saham yang Dipindai Sekaligus",
    10,
    len(all_stocks),
    100,
)
run_auto_scan = st.sidebar.button("🚀 MULAI SCANNING OTOMATIS SEKARANG")


# -----------------------------------------------------------------
# 3. FUNGSI TEKNIKAL & BBMA
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


def analyze_stock(code):
  try:
    d_exec = calculate_bbma(load_stock_data(f"{code}.JK", tf_exec))
    d_mid = calculate_bbma(load_stock_data(f"{code}.JK", tf_mid))
    d_high = calculate_bbma(load_stock_data(f"{code}.JK", tf_high))

    if d_exec.empty or d_mid.empty or d_high.empty:
      return None

    h_bull = d_high["Close"].iloc[-1] > d_high["EMA50"].iloc[-1]
    m_bull = d_mid["Close"].iloc[-1] > d_mid["BB_Middle"].iloc[-1]

    exec_row = d_exec.iloc[-1]
    # Syarat Re-Entry Buy: Harga menyentuh area MA Low dalam tren naik
    is_reentry = (
        exec_row["Low"] <= exec_row["MA10_Low"] * 1.015
        and exec_row["Close"] > exec_row["MA5_Low"]
        and exec_row["Close"] > exec_row["EMA50"]
    )

    valid_buy = is_reentry and m_bull and h_bull

    if valid_buy:
      status = "🟢 RE-ENTRY BUY VALID"
    elif h_bull and m_bull:
      status = "⏳ Menunggu Retrace"
    else:
      status = "🔴 Belum Valid"

    return {
        "Emiten": code,
        "Harga Terkini": exec_row["Close"],
        "Status Sinyal": status,
        "Valid": valid_buy,
        "DF_Exec": d_exec,
    }
  except Exception:
    return None


# -----------------------------------------------------------------
# 4. TAMPILAN UTAMA APLIKASI
# -----------------------------------------------------------------
if run_auto_scan:
  stocks_to_scan = all_stocks[:scan_limit]
  progress_bar = st.progress(0)
  status_text = st.empty()

  results = []
  total = len(stocks_to_scan)

  for i, code in enumerate(stocks_to_scan):
    status_text.text(
        f"Memindai emiten [{i+1}/{total}]: {code}.JK (Hierarki:"
        f" {tf_package}) ..."
    )
    res = analyze_stock(code)
    if res:
      results.append(res)
    progress_bar.progress((i + 1) / total)

  status_text.empty()
  progress_bar.empty()

  if results:
    df_res = pd.DataFrame(results)
    df_valid = df_res[df_res["Valid"] == True]

    st.markdown("---")
    st.subheader("🔥 HASIL FILTER: SAHAM YANG LAYAK RE-ENTRY BUY")

    if not df_valid.empty:
      st.success(
          f"Ditemukan **{len(df_valid)} emiten** dari {total} saham yang"
          " dipindai dan **paling layak** berada di posisi Re-Entry Buy saat"
          " ini!"
      )

      # Tampilkan tabel khusus emiten yang VALID
      st.dataframe(
          df_valid[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
              {"Harga Terkini": "Rp {:,.0f}"}
          ),
          use_container_width=True,
      )

      # Fitur interaktif untuk langsung melihat chart saham yang valid
      st.markdown("### 📊 Lihat Chart Detail Emiten Valid")
      selected_valid_stock = st.selectbox(
          "Pilih emiten di atas untuk melihat chart BBMA-nya:",
          df_valid["Emiten"].tolist(),
      )

      matched_row = df_valid[df_valid["Emiten"] == selected_valid_stock].iloc[0]
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
          title=f"Chart BBMA Re-Entry Buy - {selected_valid_stock}.JK",
          yaxis_title="Harga (IDR)",
          template="plotly_dark",
          height=500,
          xaxis_rangeslider_visible=False,
      )
      st.plotly_chart(fig, use_container_width=True)

    else:
      st.warning(
          "Belum ada emiten yang memenuhi kriteria mutlak Re-Entry Buy pada"
          f" batasan scan dan timeframe **{tf_package}** saat ini. Pasar sedang"
          " dalam fase *waiting mode*."
      )

    st.markdown("---")
    st.subheader("📋 Tabel Pemantauan Seluruh Emiten yang Dipindai")
    st.dataframe(
        df_res[["Emiten", "Harga Terkini", "Status Sinyal"]].style.format(
            {"Harga Terkini": "Rp {:,.0f}"}
        ),
        use_container_width=True,
    )
  else:
    st.error("Gagal menarik data pemindaian pasar.")

else:
  st.info(
      "👈 Atur paket timeframe dan jumlah batasan scan di sidebar, lalu klik"
      " tombol **🚀 MULAI SCANNING OTOMATIS SEKARANG**."
  )

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA Fuji V7 - BEI True Re-Entry Sniper",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 BBMA Fuji V7 - BEI True Re-Entry Screener")
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Penyaringan ketat sesuai hukum BBMA: **Wajib ada CSAK / CSM terlebih dahulu**, "
    "lalu harga menjemput area **MA 5/10 Low**."
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
        "🚀 Auto Screener (True Re-Entry Only)",
        "🔍 Cek Manual Emiten (Deep Analysis)",
    ],
)

tf_package = st.sidebar.selectbox(
    "Pilih Paket Hierarki Timeframe BBMA",
    ["D1 - H4 - H1", "H4 - H1 - M15"],
    index=0,
)

if tf_package == "D1 - H4 - H1":
  tf_high, tf_mid, tf_exec = "1d", "4h", "1h"
else:
  tf_high, tf_mid, tf_exec = "4h", "1h", "15m"

lookback_bars = st.sidebar.slider(
    "Batas Mundur Cek CSAK/CSM (Candle)",
    min_value=5,
    max_value=20,
    value=12,
    help="Mengecek apakah dalam N candle terakhir sudah terjadi CSAK atau CSM sebelum harga turun ke MA 5/10.",
)


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
        return calculate_bbma_fuji(df, lookback=lookback_bars)
  except Exception:
    pass
  return pd.DataFrame()


def calculate_bbma_fuji(df, lookback=12):
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

  # 1. Extreme & MHV
  df["ext_buy"] = (df["Close"] > df["lowBB"]) & (
      (df["malo5_p"] < df["lowBB"]) | (df["malo5"] < df["lowBB"])
  )
  extreme_prev_buy = df["malo5"].shift(1) < df["lowBB"].shift(1)
  is_csm_buy = df["Close"] > df["topBB"]
  df["mhv_buy"] = (
      extreme_prev_buy
      & (~is_csm_buy)
      & (df["Low"] > df["lowBB"])
      & (df["Close"] < df["midBB"])
  )

  # 2. Pemicu Arah: CSM (Momentum) & CSAK (Candle Arah Kukuh)
  df["mmt_buy"] = (df["Close"] > df["topBB"]) & (df["Open"] < df["topBB"])
  df["ali_buy"] = (df["Open"] > df["topBB"]) & (df["Close"] > df["topBB"])
  df["csm_any"] = df["mmt_buy"] | df["ali_buy"]

  df["csak_buy"] = (
      (df["Close"] > df["midBB"])
      & (df["Close"] > df["mahi5"])
      & (df["Close"] > df["mahi10"])
      & (df["Close"] > df["malo5"])
      & (df["Close"] > df["malo10"])
  )

  # 3. Validasi Riwayat Pemicu (Lookback N candle sebelumnya)
  df["prior_csm"] = (
      df["csm_any"].rolling(lookback).max().shift(1).fillna(0).astype(bool)
  )
  df["prior_csak"] = (
      df["csak_buy"].rolling(lookback).max().shift(1).fillna(0).astype(bool)
  )
  df["has_prior_trigger"] = df["prior_csm"] | df["prior_csak"]

  # 4. Re-entry Buy Sah: Wajib didahului CSAK/CSM, lalu harga menjemput MA 5/10 Low di atas Mid BB
  ma_low_upper = np.maximum(df["malo5"], df["malo10"])
  ma_low_lower = np.minimum(df["malo5"], df["malo10"])

  df["reject_malo"] = (
      df["has_prior_trigger"]
      & (df["Low"] <= ma_low_upper * 1.008)
      & (df["Close"] >= ma_low_lower * 0.99)
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
  return calculate_bbma_fuji(df_h4.dropna(), lookback=lookback_bars)


def analyze_tf_fuji(df, role_type="TF1"):
  if df.empty or len(df) < 50:
    return {
        "price": 0.0,
        "status": "DATA KURANG",
        "bias": "NEUTRAL",
        "trigger_source": "-",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "malo5": 0.0,
        "malo10": 0.0,
        "ema50": 0.0,
        "detail": "Data tidak mencukupi",
    }

  latest = df.iloc[-1]
  c_price = float(latest["Close"])
  top_bb = float(latest["topBB"])
  bot_bb = float(latest["lowBB"])
  mid_bb = float(latest["midBB"])
  malo5 = float(latest["malo5"])
  malo10 = float(latest["malo10"])
  ema50 = float(latest["ema50"])

  is_ext_buy = bool(latest["ext_buy"])
  is_mhv_buy = bool(latest["mhv_buy"])
  is_csak_buy = bool(latest["csak_buy"])
  is_mmt_buy = bool(latest["csm_any"])
  is_re_buy = bool(latest["reject_malo"])

  prior_csm = bool(latest["prior_csm"])
  prior_csak = bool(latest["prior_csak"])

  if prior_csm and prior_csak:
    trigger_src = "Pasca CSM & CSAK"
  elif prior_csm:
    trigger_src = "Pasca CSM (Momentum)"
  elif prior_csak:
    trigger_src = "Pasca CSAK (Arah Kukuh)"
  else:
    trigger_src = "Belum Ada Pemicu"

  status = "-"
  bias = "NEUTRAL"
  detail = "-"

  if is_re_buy:
    status = f"🔥 RE-ENTRY VALID ({trigger_src})"
    bias = "BUY"
    detail = (
        f"Saham sudah mencetak {trigger_src}, lalu harga koreksi sehat"
        " menjemput MA 5/10 Low di atas Mid BB."
    )
  elif is_mmt_buy:
    status = "🚀 MOMENTUM BUY (CSM Aktif)"
    bias = "BUY"
    detail = "Candle baru menembus Top BB. Tunggu pullback ke MA 5/10 Low."
  elif is_csak_buy:
    status = "📈 CSAK BUY (Pecah Arah)"
    bias = "BUY"
    detail = "Candle menembus Mid BB & MA High. Siap pantau area Re-entry."
  elif is_mhv_buy or is_ext_buy:
    status = "⚡ EXTREME / MHV BUY"
    bias = "BUY"
    detail = "Terdeteksi Extreme/MHV di Low BB (menunggu konfirmasi CSAK)."
  elif c_price > mid_bb:
    status = "⏳ BULLISH (Tanpa Trigger Re-Entry)"
    bias = "BUY"
    detail = (
        "Harga di atas Mid BB, namun belum memenuhi syarat pullback pasca"
        " CSAK/CSM."
    )
  else:
    status = "⚠️ BEARISH / DI BAWAH MID BB"
    bias = "SELL"
    detail = "Struktur harga masih berada di bawah Mid BB."

  return {
      "price": c_price,
      "status": status,
      "bias": bias,
      "trigger_source": trigger_src,
      "top_bb": top_bb,
      "bot_bb": bot_bb,
      "mid_bb": mid_bb,
      "malo5": malo5,
      "malo10": malo10,
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

    # KUNCI FILTER KETAT:
    # 1. TF Utama (TF1) dan TF Menengah (TF2) wajib searah BUY
    # 2. TF Eksekusi (TF3) atau TF Menengah (TF2) wajib sedang status "RE-ENTRY VALID" (sudah ada CSAK/CSM sebelumnya)
    is_true_reentry = (
        res_tf1["bias"] == "BUY"
        and res_tf2["bias"] == "BUY"
        and (
            "RE-ENTRY VALID" in res_tf3["status"]
            or "RE-ENTRY VALID" in res_tf2["status"]
        )
    )

    if not is_true_reentry:
      return None

    # Tentukan asal trigger dari TF yang sedang Re-Entry
    active_trigger = (
        res_tf3["trigger_source"]
        if "RE-ENTRY VALID" in res_tf3["status"]
        else res_tf2["trigger_source"]
    )

    # Zona Entry Presisi (Rentang antara MA 5 Low dan MA 10 Low di TF Eksekusi)
    entry_low = min(res_tf3["malo5"], res_tf3["malo10"])
    entry_high = max(res_tf3["malo5"], res_tf3["malo10"])

    return {
        "Emiten": code,
        "Harga Terkini": res_tf3["price"],
        "Nilai Transaksi": f"Rp {val_trx:,.0f}",
        "Zona Entry (MA 5/10 Low)": (
            f"Rp {entry_low:,.0f} - Rp {entry_high:,.0f}"
        ),
        "Dasar Re-Entry": active_trigger,
        "Status Setup": "🎯 RE-ENTRY VALID SIAP BELI",
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
if mode == "🚀 Auto Screener (True Re-Entry Only)":
  st.subheader(
      "🚀 Auto Screener - Emiten Valid Re-Entry (Pasca CSAK / CSM)"
  )
  st.markdown(
      "Menyaring emiten BEI yang **sudah mencetak arah (CSAK/CSM)** dan saat ini"
      " harganya sedang **menjemput area MA 5/10 Low**."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 80
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING TRUE RE-ENTRY")

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
      st.subheader("🔥 DAFTAR SAHAM LAYAK RE-ENTRY (VALIDATED)")
      st.success(
          f"Ditemukan **{len(df_res)} emiten** yang valid memenuhi syarat"
          " Re-Entry pasca CSAK/CSM!"
      )

      st.dataframe(
          df_res[
              [
                  "Emiten",
                  "Harga Terkini",
                  "Nilai Transaksi",
                  "Zona Entry (MA 5/10 Low)",
                  "Dasar Re-Entry",
                  "Status Setup",
              ]
          ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
          use_container_width=True,
      )

      selected_valid = st.selectbox(
          "Pilih emiten di atas untuk melihat detail chart & riwayat pemicunya:",
          df_res["Emiten"].tolist(),
      )
      matched_row = df_res[df_res["Emiten"] == selected_valid].iloc[0]
      d_target = matched_row["DF_Exec"]

      with st.expander(
          f"📌 Rincian Struktur Multi-Timeframe: {selected_valid}",
          expanded=True,
      ):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
          st.markdown(f"**TF Utama ({tf_high.upper()})**")
          st.info(
              f"**Status:** {matched_row['TF1']['status']}\n\n•"
              f" {matched_row['TF1']['detail']}"
          )
        with rc2:
          st.markdown(f"**TF Menengah ({tf_mid.upper()})**")
          st.info(
              f"**Status:** {matched_row['TF2']['status']}\n\n•"
              f" {matched_row['TF2']['detail']}"
          )
        with rc3:
          st.markdown(f"**TF Eksekusi ({tf_exec.upper()})**")
          st.info(
              f"**Status:** {matched_row['TF3']['status']}\n\n•"
              f" {matched_row['TF3']['detail']}"
          )

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
              y=d_target["malo5"],
              name="MA 5 Low",
              line=dict(color="blue", width=1.2),
          )
      )
      fig.add_trace(
          go.Scatter(
              x=d_target.index,
              y=d_target["malo10"],
              name="MA 10 Low",
              line=dict(color="cyan", width=1.2),
          )
      )

      fig.update_layout(
          title=(
              f"Chart Re-Entry Valid - {selected_valid}.JK"
              f" ({tf_exec.upper()})"
          ),
          yaxis_title="Harga (IDR)",
          template="plotly_dark",
          height=500,
          xaxis_rangeslider_visible=False,
      )
      st.plotly_chart(fig, use_container_width=True)

    else:
      st.warning(
          "⚠️ Tidak ditemukan emiten yang valid Re-Entry pasca CSAK/CSM pada"
          " batch pemindaian ini."
      )
  else:
    st.info(
        "👈 Atur parameter di sidebar, lalu klik **⚡ MULAI SCANNING TRUE"
        " RE-ENTRY** untuk mencari saham yang sah secara hukum BBMA."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL EMITEN
# -----------------------------------------------------------------
else:
  st.subheader(
      "🔍 Cek Manual Emiten - Verifikasi Hukum Re-Entry BBMA"
  )
  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(f"Memverifikasi struktur BBMA {selected_stock}.JK ..."):
    if tf_high == "4h":
      df_h1_temp = load_stock_data(f"{selected_stock}.JK", "1h")
      d_high_m = resample_to_h4(df_h1_temp)
    else:
      d_high_m = load_stock_data(f"{selected_stock}.JK", tf_high)

    d_mid_m = load_stock_data(f"{selected_stock}.JK", tf_mid)
    d_exec_m = load_stock_data(f"{selected_stock}.JK", tf_exec)

    res_tf1_m = analyze_tf_fuji(d_high_m, role_type="TF1")
    res_tf2_m = analyze_tf_fuji(d_mid_m, role_type="TF2")
    res_tf3_m = analyze_tf_fuji(d_exec_m, role_type="TF3")

    res_manual = evaluate_pure_bbma_screener(
        selected_stock, tf_high, tf_mid, tf_exec
    )

  if d_exec_m.empty:
    st.warning(
        f"⚠️ Gagal memuat data untuk **{selected_stock}.JK**. Periksa likuiditas"
        " saham tersebut."
    )
  else:
    latest_price = d_exec_m["Close"].iloc[-1]
    prev_price = (
        d_exec_m["Close"].iloc[-2] if len(d_exec_m) > 1 else latest_price
    )
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100 if prev_price > 0 else 0

    entry_low_m = min(res_tf3_m["malo5"], res_tf3_m["malo10"])
    entry_high_m = max(res_tf3_m["malo5"], res_tf3_m["malo10"])

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.5])
    c1.metric(
        f"Harga {selected_stock}",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric(
        "Nilai Transaksi",
        f"Rp {latest_price * d_exec_m['Volume'].iloc[-1]:,.0f}",
    )
    c3.metric(
        "Zona MA 5/10 Low",
        f"Rp {entry_low_m:,.0f} - {entry_high_m:,.0f}",
    )

    status_text = (
        res_manual["Status Setup"]
        if res_manual
        else "Belum Valid (Tunggu CSAK/CSM + Pullback)"
    )
    c4.markdown(
        f"""
        <div style="background-color: #262730; padding: 10px 14px; border-radius: 4px; border: 1px solid #41424C;">
            <p style="color: #9N9N9N; font-size: 12px; margin-bottom: 2px;">Keputusan Sistem</p>
            <p style="color: #FFFFFF; font-size: 15px; font-weight: 600; margin: 0; word-break: break-word;">{status_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 📌 Rincian Alasan & Riwayat Pemicu Multi-Timeframe")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
      st.markdown(f"**TF Utama ({tf_high.upper()})**")
      st.info(
          f"**Status:** {res_tf1_m['status']}\n\n• {res_tf1_m['detail']}"
      )
    with rc2:
      st.markdown(f"**TF Menengah ({tf_mid.upper()})**")
      st.info(
          f"**Status:** {res_tf2_m['status']}\n\n• {res_tf2_m['detail']}"
      )
    with rc3:
      st.markdown(f"**TF Eksekusi ({tf_exec.upper()})**")
      st.info(
          f"**Status:** {res_tf3_m['status']}\n\n• {res_tf3_m['detail']}"
      )

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d_exec_m.index,
            open=d_exec_m["Open"],
            high=d_exec_m["High"],
            low=d_exec_m["Low"],
            close=d_exec_m["Close"],
            name="Harga",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec_m.index,
            y=d_exec_m["topBB"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec_m.index,
            y=d_exec_m["midBB"],
            name="Mid BB",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec_m.index,
            y=d_exec_m["lowBB"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec_m.index,
            y=d_exec_m["malo5"],
            name="MA 5 Low",
            line=dict(color="blue", width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec_m.index,
            y=d_exec_m["malo10"],
            name="MA 10 Low",
            line=dict(color="cyan", width=1.2),
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

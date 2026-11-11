from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA Fuji V7 Pro BEI Screener", page_icon="📈", layout="wide"
)

st.title(
    "📈 BBMA Fuji V7 Pro - Multi-Timeframe Re-Entry Buy Screener (944+ Saham"
    " BEI)"
)
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem validasi hierarki ketat khusus **Re-Entry Buy** (TF1 ➔ TF2 ➔ TF3)"
    " menggunakan logika **Fuji V7 (Oma Ally)** untuk emiten Bursa Efek"
    " Indonesia."
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
        "🚀 Auto Screener (Cari Re-Entry Buy Ketat)",
        "🔍 Cek Manual (Single Stock Pro)",
    ],
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
      df = df.dropna(subset=["Open", "High", "Low", "Close"])
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

  # Shift untuk data sebelumnya ([1])
  df["mahi5_p"] = df["mahi5"].shift(1)
  df["malo5_p"] = df["malo5"].shift(1)
  df["topBB_p"] = df["topBB"].shift(1)
  df["lowBB_p"] = df["lowBB"].shift(1)

  # --- PORTING LOGIKA SINYAL DARI FUJI V7 PINE SCRIPT ---
  # 1. Extreme
  df["ext_sell"] = (df["Close"] < df["topBB"]) & (
      (df["mahi5_p"] > df["topBB"]) | (df["mahi5"] > df["topBB"])
  )
  df["ext_buy"] = (df["Close"] > df["lowBB"]) & (
      (df["malo5_p"] < df["lowBB"]) | (df["malo5"] < df["lowBB"])
  )

  # 2. CSM / Momentum (mmt) & CSMKuat (ali)
  df["mmt_sell"] = (df["Close"] < df["lowBB"]) & (df["Open"] > df["lowBB"])
  df["mmt_buy"] = (df["Close"] > df["topBB"]) & (df["Open"] < df["topBB"])
  df["ali_sell"] = (df["Open"] < df["lowBB"]) & (df["Close"] < df["lowBB"])
  df["ali_buy"] = (df["Open"] > df["topBB"]) & (df["Close"] > df["topBB"])

  # 3. MHV
  extreme_prev_sell = df["mahi5"].shift(1) > df["topBB"].shift(1)
  extreme_prev_buy = df["malo5"].shift(1) < df["lowBB"].shift(1)
  is_csm_sell = df["Close"] < df["lowBB"]
  is_csm_buy = df["Close"] > df["topBB"]
  df["mhv_sell"] = (
      extreme_prev_sell
      & (~is_csm_sell)
      & (df["High"] < df["topBB"])
      & (df["Close"] > df["midBB"])
  )
  df["mhv_buy"] = (
      extreme_prev_buy
      & (~is_csm_buy)
      & (df["Low"] > df["lowBB"])
      & (df["Close"] < df["midBB"])
  )

  # 4. CSAK
  df["csak_sell"] = (
      (df["Close"] < df["midBB"])
      & (df["Close"] < df["mahi5"])
      & (df["Close"] < df["mahi10"])
      & (df["Close"] < df["malo5"])
      & (df["Close"] < df["malo10"])
  )
  df["csak_buy"] = (
      (df["Close"] > df["midBB"])
      & (df["Close"] > df["mahi5"])
      & (df["Close"] > df["mahi10"])
      & (df["Close"] > df["malo5"])
      & (df["Close"] > df["malo10"])
  )

  # 5. Reentry (reject_mahi / reject_malo)
  df["reject_mahi"] = (
      (df["High"] > df["mahi5"])
      & (df["Close"] < df["mahi5"])
      & (df["Close"] < df["mahi10"])
      & (df["Close"] < df["midBB"])
      & (df["mahi5"] < df["midBB"])
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
        "detail": "Data tidak mencukupi untuk indikator lengkap",
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

  # 1. ATURAN TF1: Acuan Utama Re-Entry Buy / Bullish
  if role_type == "TF1":
    if is_re_buy or (
        c_price >= bot_bb and c_price <= latest["malo10"] * 1.025
    ):
      status = "RE-ENTRY BUY VALID (Acuan Utama)"
      bias = "BUY"
      detail = "Harga melakukan penolakan valid di area MA Low (Fuji V7)."
    elif is_mhv_buy or is_ext_buy:
      status = "EXTREME / MHV BUY (Acuan Utama)"
      bias = "BUY"
      detail = "Terdeteksi Extreme/MHV Buy di Low BB pada TF1."
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menunggu Re-Entry Buy)"
      bias = "BUY"
      detail = "Harga di atas Mid BB, menunggu pullback ke MA Low."
    else:
      status = "BEARISH / TIDAK VALID"
      bias = "SELL"
      detail = "TF1 berada di bawah Mid BB (Tidak memenuhi syarat Buy)."

  # 2. ATURAN TF2: Konfirmasi
  elif role_type == "TF2":
    if is_re_buy or is_ext_buy or is_mhv_buy or is_mmt_buy:
      status = "KONFIRMASI BUY (Valid Fuji V7)"
      bias = "BUY"
      detail = "TF2 mengonfirmasi lewat Re-Entry/Extreme/MHV/Momentum Buy."
    elif is_csak_buy or c_price > mid_bb:
      status = "TF2 BULLISH / CSAK BUY"
      bias = "BUY"
      detail = "Struktur TF2 membentuk CSAK Buy atau berada di atas Mid BB."
    else:
      status = "TF2 BELUM KONFIRMASI"
      bias = "SELL"
      detail = "TF2 belum menunjukkan konfirmasi arah Buy."

  # 3. ATURAN TF3: Trigger Eksekusi Akhir
  elif role_type == "TF3":
    if is_mmt_buy:
      status = "TRIGGER MOMENTUM BUY"
      bias = "BUY"
      detail = "TF3 membentuk Momentum Buy menembus Top BB."
    elif is_csak_buy:
      status = "TRIGGER CSAK BUY"
      bias = "BUY"
      detail = "TF3 memicu CSAK Buy melewati Mid BB."
    elif is_re_buy or is_ext_buy or is_mhv_buy:
      status = "TRIGGER RE-ENTRY / EXTREME BUY"
      bias = "BUY"
      detail = "TF3 siap eksekusi Re-Entry Buy di area MA Low."
    elif c_price > mid_bb:
      status = "TF3 BULLISH MENGALIR"
      bias = "BUY"
      detail = "Kondisi TF3 mengalir searah Buy."
    else:
      status = "TF3 BELUM ADA TRIGGER"
      bias = "SELL"
      detail = "TF3 belum memunculkan trigger eksekusi Buy."

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


def evaluate_pro_buy_combo(code, tf_high_code, tf_mid_code, tf_exec_code):
  try:
    if tf_high_code == "4h":
      df_h1 = load_stock_data(f"{code}.JK", "1h")
      d_high = resample_to_h4(df_h1)
    else:
      d_high = load_stock_data(f"{code}.JK", tf_high_code)

    d_mid = load_stock_data(f"{code}.JK", tf_mid_code)
    d_exec = load_stock_data(f"{code}.JK", tf_exec_code)

    if d_high.empty or d_mid.empty or d_exec.empty:
      return None

    res_tf1 = analyze_tf_fuji(d_high, role_type="TF1")
    res_tf2 = analyze_tf_fuji(d_mid, role_type="TF2")
    res_tf3 = analyze_tf_fuji(d_exec, role_type="TF3")

    is_valid_buy = (
        res_tf1["bias"] == "BUY"
        and res_tf2["bias"] == "BUY"
        and res_tf3["bias"] == "BUY"
        and "RE-ENTRY" in res_tf1["status"]
    )

    entry_zone = (res_tf3["bot_bb"] + res_tf3["mid_bb"]) / 2

    if is_valid_buy:
      status_label = "🔥 PRO RE-ENTRY BUY VALID (Fuji V7)"
    elif res_tf1["bias"] == "BUY" and res_tf2["bias"] == "BUY":
      status_label = "⏳ Menunggu Trigger TF3"
    else:
      status_label = "❌ Belum Valid"

    return {
        "Emiten": code,
        "Harga Terkini": res_tf3["price"],
        "Estimasi Area Entry": f"Rp {entry_zone:,.0f}",
        "Status Sinyal": status_label,
        "Valid": is_valid_buy,
        "TF1": res_tf1,
        "TF2": res_tf2,
        "TF3": res_tf3,
        "DF_Exec": d_exec,
    }
  except Exception:
    return None


# -----------------------------------------------------------------
# MODE 1: AUTO SCREENER
# -----------------------------------------------------------------
if mode == "🚀 Auto Screener (Cari Re-Entry Buy Ketat)":
  st.subheader(
      "🚀 Auto Screener - Filter Ketat Pro Re-Entry Buy (Logika Fuji V7)"
  )
  st.markdown(
      "Sistem memindai emiten BEI secara otomatis untuk menyaring mana saja"
      " yang memenuhi syarat hierarki ketat **TF1 (Re-Entry) ➔ TF2 (Konfirmasi)"
      f" ➔ TF3 (Trigger)** menggunakan paket **{tf_package}**."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 60
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING FUJI V7 BUY")

  if run_scan:
    stocks_to_scan = all_stocks[:scan_limit]
    prog_bar = st.progress(0)
    txt_status = st.empty()

    results = []
    total = len(stocks_to_scan)

    for i, code in enumerate(stocks_to_scan):
      txt_status.text(f"Memindai emiten [{i+1}/{total}]: {code}.JK ...")
      res = evaluate_pro_buy_combo(code, tf_high, tf_mid, tf_exec)
      if res:
        results.append(res)
      prog_bar.progress((i + 1) / total)

    txt_status.empty()
    prog_bar.empty()

    if results:
      df_res = pd.DataFrame(results)
      df_valid = df_res[df_res["Valid"] == True]

      st.markdown("---")
      st.subheader("🔥 HASIL FILTER: SAHAM FUJI V7 VALID RE-ENTRY BUY")

      if not df_valid.empty:
        st.success(
            f"Ditemukan **{len(df_valid)} emiten** dengan setup Pro Re-Entry Buy"
            " Fuji V7 yang valid!"
        )
        st.dataframe(
            df_valid[
                [
                    "Emiten",
                    "Harga Terkini",
                    "Estimasi Area Entry",
                    "Status Sinyal",
                ]
            ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
            use_container_width=True,
        )

        selected_valid = st.selectbox(
            "Pilih emiten valid di atas untuk melihat detail chart:",
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
            title=f"Chart Eksekusi Fuji V7 - {selected_valid}.JK",
            yaxis_title="Harga (IDR)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      else:
        st.warning(
            "Belum ada emiten yang memenuhi kriteria mutlak Pro Re-Entry Buy"
            " Fuji V7 saat ini."
        )

      st.markdown("---")
      st.subheader("📋 Seluruh Hasil Pemantauan Pasar")
      st.dataframe(
          df_res[
              ["Emiten", "Harga Terkini", "Estimasi Area Entry", "Status Sinyal"]
          ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
          use_container_width=True,
      )
    else:
      st.error("Gagal menarik data pemindaian pasar.")
  else:
    st.info(
        "👈 Atur batasan scan di sidebar, lalu klik **⚡ MULAI SCANNING FUJI V7"
        " BUY** untuk memfilter otomatis."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL
# -----------------------------------------------------------------
else:
  st.subheader(
      "🔍 Cek Manual Pro - Analisis Hirarki Re-Entry Buy (Fuji V7 Logic)"
  )
  st.markdown(
      "Pilih emiten apa saja untuk melihat rincian validasi per timeframe"
      " berdasarkan aturan ketat Re-Entry Buy Fuji V7."
  )

  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(
      f"Memuat data analisis Fuji V7 untuk {selected_stock}.JK ..."
  ):
    res = evaluate_pro_buy_combo(selected_stock, tf_high, tf_mid, tf_exec)

  if not res:
    st.warning(
        f"⚠️ Gagal memuat data untuk **{selected_stock}.JK**. Periksa likuiditas"
        " saham tersebut."
    )
  else:
    d_exec = res["DF_Exec"]
    latest_price = res["Harga Terkini"]
    prev_price = d_exec["Close"].iloc[-2] if len(d_exec) > 1 else latest_price
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100 if prev_price > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Harga {selected_stock}",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric("TF Eksekusi (TF3)", tf_exec.upper())
    c3.metric("TF Konfirmasi (TF2)", tf_mid.upper())
    c4.metric("TF Acuan Re-Entry (TF1)", tf_high.upper())

    st.markdown("---")
    st.subheader(f"🎯 Status Sinyal Fuji V7: {res['Status Sinyal']}")
    st.markdown(f"📍 **Estimasi Area Titik Entry:** **{res['Estimasi Area Entry']}**")

    if res["Valid"]:
      st.success(
          f"Saham **{selected_stock}** saat ini memenuhi **Pro Setup Valid"
          f" (Re-Entry Buy)** secara sempurna berdasarkan hirarki {tf_package}"
          " (Fuji V7)!"
      )
    else:
      st.info(
          f"Saham **{selected_stock}** belum memenuhi seluruh kriteria hirarki"
          " Re-Entry Buy Fuji V7 pada saat ini."
      )

    st.markdown("---")
    st.subheader(
        "📌 Matriks Validasi Hirarki Pro Fuji V7 (TF1 ➔ TF2 ➔ TF3)"
    )

    matrix_rows = [
        {
            "Peran Hirarki": f"TF1 (Acuan Re-Entry - {tf_high.upper()})",
            "Status / Kondisi": res["TF1"]["status"],
            "Arah Bias": res["TF1"]["bias"],
            "Top BB": f"Rp {res['TF1']['top_bb']:,.2f}",
            "Mid BB": f"Rp {res['TF1']['mid_bb']:,.2f}",
            "EMA 50": f"Rp {res['TF1']['ema50']:,.2f}",
            "Bottom BB": f"Rp {res['TF1']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF1"]["detail"],
        },
        {
            "Peran Hirarki": f"TF2 (Konfirmasi - {tf_mid.upper()})",
            "Status / Kondisi": res["TF2"]["status"],
            "Arah Bias": res["TF2"]["bias"],
            "Top BB": f"Rp {res['TF2']['top_bb']:,.2f}",
            "Mid BB": f"Rp {res['TF2']['mid_bb']:,.2f}",
            "EMA 50": f"Rp {res['TF2']['ema50']:,.2f}",
            "Bottom BB": f"Rp {res['TF2']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF2"]["detail"],
        },
        {
            "Peran Hirarki": f"TF3 (Trigger Eksekusi Akhir - {tf_exec.upper()})",
            "Status / Kondisi": res["TF3"]["status"],
            "Arah Bias": res["TF3"]["bias"],
            "Top BB": f"Rp {res['TF3']['top_bb']:,.2f}",
            "Mid BB": f"Rp {res['TF3']['mid_bb']:,.2f}",
            "EMA 50": f"Rp {res['TF3']['ema50']:,.2f}",
            "Bottom BB": f"Rp {res['TF3']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF3"]["detail"],
        },
    ]
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

    st.markdown("---")

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
        title=(
            f"Analisis Detail Fuji V7 BBMA - {selected_stock}.JK ({tf_exec})"
        ),
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

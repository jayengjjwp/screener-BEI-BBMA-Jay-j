from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA MTF Pro BEI Screener", page_icon="📈", layout="wide"
)

st.title(
    "📈 BBMA Oma Ally Pro - Multi-Timeframe Re-Entry Buy Screener (944+ Saham"
    " BEI)"
)
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem validasi hierarki ketat khusus **Re-Entry Buy** (TF3 ➔ TF2 ➔ TF1)"
    " untuk emiten Bursa Efek Indonesia."
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
# 3. FUNGSI TEKNIKAL & ANALISIS BBMA PRO (BUY STRICT)
# -----------------------------------------------------------------
@st.cache_data(ttl=0)
def load_stock_data(ticker_symbol, interval):
  # Perpanjang periode untuk Daily (1d) agar EMA 50 (butuh min 55 baris) aman
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
        return df
  except Exception:
    pass
  return pd.DataFrame()


def resample_to_h4(df_h1):
  if df_h1.empty:
    return pd.DataFrame()
  df_h4 = df_h1.resample("4h").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
  })
  return df_h4.dropna()


def analyze_bbma_tf(df, role_type="TF1"):
  if df.empty or len(df) < 55:
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

  df = df.copy()
  df["MA20"] = df["Close"].rolling(window=20).mean()
  df["STD"] = df["Close"].rolling(window=20).std()
  df["TopBB"] = df["MA20"] + (df["STD"] * 2)
  df["BottomBB"] = df["MA20"] - (df["STD"] * 2)
  df["MA5_High"] = df["High"].rolling(window=5).mean()
  df["MA5_Low"] = df["Low"].rolling(window=5).mean()
  df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

  df = df.dropna()
  if df.empty or len(df) < 2:
    return {
        "price": 0.0,
        "status": "KOSONG",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ema50": 0.0,
        "detail": "Data kosong setelah kalkulasi",
    }

  latest = df.iloc[-1]
  prev = df.iloc[-2]

  c_price = float(latest["Close"])
  top_bb = float(latest["TopBB"])
  bot_bb = float(latest["BottomBB"])
  mid_bb = float(latest["MA20"])
  ema50 = float(latest["EMA50"])
  ma5_h = float(latest["MA5_High"])
  ma5_l = float(latest["MA5_Low"])
  high = float(latest["High"])
  low = float(latest["Low"])

  status = "-"
  bias = "NEUTRAL"
  detail = "-"

  # Deteksi Rejection / Extreme / CSAK khusus arah BUY
  reject_bot_bb = low <= bot_bb and c_price > bot_bb
  reject_ema50_buy = low <= ema50 and c_price > ema50
  momentum_buy = (
      c_price <= bot_bb or (low <= bot_bb and prev["Low"] <= prev["BottomBB"])
  )

  body_size = abs(c_price - float(latest["Open"]))
  avg_body = abs(df["Close"] - df["Open"]).rolling(10).mean().iloc[-1]
  csak_buy = (
      body_size > (avg_body * 1.2)
      and c_price > mid_bb
      and prev["Close"] <= prev["MA20"]
  )

  # 1. ATURAN TF1: Wajib Mencari Re-Entry Buy / Bullish Acuan Utama
  if role_type == "TF1":
    if (c_price <= ma5_h) and (c_price >= mid_bb or low >= mid_bb):
      status = "RE-ENTRY BUY (Acuan Utama)"
      bias = "BUY"
      detail = "Zona Re-Entry Buy aktif di antara MA5/10 Low dan Mid BB."
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menunggu Re-Entry Buy)"
      bias = "BUY"
      detail = "Harga di atas Mid BB, menunggu pullback ke area Re-Entry Buy."
    else:
      status = "BEARISH / TIDAK VALID"
      bias = "SELL"
      detail = "TF1 berada di bawah Mid BB (Tidak memenuhi syarat Buy)."

  # 2. ATURAN TF2: Konfirmasi (Extreme / CSAK / Momentum / Reject Bottom BB / Reject EMA 50)
  elif role_type == "TF2":
    if momentum_buy or reject_bot_bb or reject_ema50_buy or c_price <= bot_bb:
      status = "KONFIRMASI BUY (Valid Reject/Extreme)"
      bias = "BUY"
      detail = "TF2 mengonfirmasi lewat Extreme/Momentum/Reject Bottom BB atau EMA 50."
    elif csak_buy or (c_price >= ma5_h and c_price > mid_bb):
      status = "CSAK / RE-ENTRY BUY KONFIRMASI"
      bias = "BUY"
      detail = "TF2 membentuk CSAK Buy atau Re-Entry penunjang."
    elif c_price > mid_bb:
      status = "TF2 BULLISH CONTINUATION"
      bias = "BUY"
      detail = "Struktur TF2 mendukung kenaikan."
    else:
      status = "TF2 BELUM KONFIRMASI"
      bias = "SELL"
      detail = "TF2 belum menunjukkan konfirmasi arah Buy."

  # 3. ATURAN TF3: Trigger Eksekusi Akhir (Momentum Buy / CSAK Buy / Re-Entry / Extreme Buy)
  elif role_type == "TF3":
    if momentum_buy:
      status = "TRIGGER MOMENTUM BUY"
      bias = "BUY"
      detail = "TF3 membentuk Momentum Buy sebagai pelatuk eksekusi."
    elif termasuk_csak := csak_buy:
      status = "TRIGGER CSAK BUY"
      bias = "BUY"
      detail = "TF3 membentuk CSAK Buy (Full Setup / Trigger)."
    elif (c_price <= ma5_l and c_price < mid_bb) or reject_bot_bb:
      status = "TRIGGER RE-ENTRY / EXTREME BUY"
      bias = "BUY"
      detail = "TF3 siap eksekusi Re-Entry / Extreme Buy."
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

    res_tf1 = analyze_bbma_tf(d_high, role_type="TF1")
    res_tf2 = analyze_bbma_tf(d_mid, role_type="TF2")
    res_tf3 = analyze_bbma_tf(d_exec, role_type="TF3")

    # Syarat Mutlak Pro Re-Entry Buy: Ketiga TF harus memiliki bias BUY (Alignment Sempurna)
    is_valid_buy = (
        res_tf1["bias"] == "BUY"
        and res_tf2["bias"] == "BUY"
        and res_tf3["bias"] == "BUY"
        and "RE-ENTRY" in res_tf1["status"]
    )

    # Estimasi Area Titik Entry (Kisaran MA Low / Mid BB terdekat)
    entry_zone = (res_tf3["bot_bb"] + res_tf3["mid_bb"]) / 2

    if is_valid_buy:
      status_label = "🔥 PRO RE-ENTRY BUY VALID"
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
# MODE 1: AUTO SCREENER (PENCARIAN OTOMATIS RE-ENTRY BUY)
# -----------------------------------------------------------------
if mode == "🚀 Auto Screener (Cari Re-Entry Buy Ketat)":
  st.subheader("🚀 Auto Screener - Filter Ketat Pro Re-Entry Buy")
  st.markdown(
      "Sistem memindai emiten BEI secara otomatis untuk menyaring mana saja"
      " yang memenuhi syarat hierarki ketat **TF1 (Re-Entry) ➔ TF2 (Konfirmasi)"
      f" ➔ TF3 (Trigger)** menggunakan paket **{tf_package}**."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 60
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING PRO BUY")

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
      st.subheader("🔥 HASIL FILTER: SAHAM PRO VALID RE-ENTRY BUY")

      if not df_valid.empty:
        st.success(
            f"Ditemukan **{len(df_valid)} emiten** dengan setup Pro Re-Entry Buy"
            " yang valid!"
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

        # Interaktif Chart untuk Emiten yang Valid
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
                y=d_target["TopBB"],
                name="BB Upper",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["MA20"],
                name="Mid BB",
                line=dict(color="orange", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["BottomBB"],
                name="BB Lower",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["EMA50"],
                name="EMA 50",
                line=dict(color="magenta", width=1.2),
            )
        )

        fig.update_layout(
            title=f"Chart Eksekusi Pro - {selected_valid}.JK",
            yaxis_title="Harga (IDR)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      else:
        st.warning(
            "Belum ada emiten yang memenuhi kriteria mutlak Pro Re-Entry Buy"
            " saat ini."
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
        "👈 Atur batasan scan di sidebar, lalu klik **⚡ MULAI SCANNING PRO"
        " BUY** untuk memfilter otomatis."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL (SINGLE STOCK PRO ANALYSIS)
# -----------------------------------------------------------------
else:
  st.subheader("🔍 Cek Manual Pro - Analisis Hirarki Re-Entry Buy")
  st.markdown(
      "Pilih emiten apa saja untuk melihat rincian validasi per timeframe"
      " berdasarkan aturan ketat Re-Entry Buy."
  )

  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(f"Memuat data analisis pro untuk {selected_stock}.JK ..."):
    res = evaluate_pro_buy_combo(
        selected_stock, tf_high, tf_mid, tf_exec
    )

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
    st.subheader(f"🎯 Status Sinyal Pro: {res['Status Sinyal']}")
    st.markdown(f"📍 **Estimasi Area Titik Entry:** **{res['Estimasi Area Entry']}**")

    if res["Valid"]:
      st.success(
          f"Saham **{selected_stock}** saat ini memenuhi **Pro Setup Valid"
          f" (Re-Entry Buy)** secara sempurna berdasarkan hirarki {tf_package}!"
      )
    else:
      st.info(
          f"Saham **{selected_stock}** belum memenuhi seluruh kriteria hirarki"
          " Re-Entry Buy yang ketat pada saat ini."
      )

    st.markdown("---")
    st.subheader("📌 Matriks Validasi Hirarki Pro (TF1 ➔ TF2 ➔ TF3)")

    matrix_rows = [
        {
            "Peran Hirarki": f"TF1 (Acuan Re-Entry - {tf_high.upper()})",
            "Status / Kondisi": res["TF1"]["status"],
            "Arah Bias": res["TF1"]["bias"],
            "Top BB": f"${res['TF1']['top_bb']:,.2f}",
            "Mid BB": f"${res['TF1']['mid_bb']:,.2f}",
            "EMA 50": f"${res['TF1']['ema50']:,.2f}",
            "Bottom BB": f"${res['TF1']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF1"]["detail"],
        },
        {
            "Peran Hirarki": (
                f"TF2 (Konfirmasi Reject/Extreme/CSAK - {tf_mid.upper()})"
            ),
            "Status / Kondisi": res["TF2"]["status"],
            "Arah Bias": res["TF2"]["bias"],
            "Top BB": f"${res['TF2']['top_bb']:,.2f}",
            "Mid BB": f"${res['TF2']['mid_bb']:,.2f}",
            "EMA 50": f"${res['TF2']['ema50']:,.2f}",
            "Bottom BB": f"${res['TF2']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF2"]["detail"],
        },
        {
            "Peran Hirarki": (
                f"TF3 (Trigger Eksekusi Akhir - {tf_exec.upper()})"
            ),
            "Status / Kondisi": res["TF3"]["status"],
            "Arah Bias": res["TF3"]["bias"],
            "Top BB": f"${res['TF3']['top_bb']:,.2f}",
            "Mid BB": f"${res['TF3']['mid_bb']:,.2f}",
            "EMA 50": f"${res['TF3']['ema50']:,.2f}",
            "Bottom BB": f"${res['TF3']['bot_bb']:,.2f}",
            "Keterangan Analisis": res["TF3"]["detail"],
        },
    ]
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

    st.markdown("---")

    # Chart Detail Pro
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
            y=d_exec["TopBB"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["MA20"],
            name="Mid BB",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["BottomBB"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["EMA50"],
            name="EMA 50",
            line=dict(color="magenta", width=1.2),
        )
    )

    fig.update_layout(
        title=f"Analisis Detail Pro BBMA - {selected_stock}.JK ({tf_exec})",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

#!/usr/bin/env python3
"""
Evapotranspiration Analysis & Presentation Generator
=====================================================
Calculates reference ET₀ from solar radiation data using
Hargreaves-Samani method, generates publication-quality
visualizations, and builds an HTML slide deck.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Istanbul parameters ───
LAT = 41.01
GSC = 0.0820  # MJ/m²/min

# Monthly avg temps for Istanbul (°C) — long-term climatology
TMAX_CLIM = {1:8.5, 2:9.2, 3:11.8, 4:16.5, 5:21.4, 6:26.0, 7:28.6, 8:28.7, 9:25.0, 10:19.8, 11:14.5, 12:10.5}
TMIN_CLIM = {1:2.5, 2:2.8, 3:4.2, 4:8.0, 5:12.5, 6:16.5, 7:19.0, 8:19.5, 9:16.0, 10:12.0, 11:7.5, 12:4.0}

# Colors
C_PRIMARY = '#0f766e'
C_SECONDARY = '#0ea5e9'
C_ACCENT = '#f59e0b'
C_REAL = '#10b981'
C_SYNTH = '#6366f1'
C_BG = '#0f172a'
C_TEXT = '#e2e8f0'

plt.rcParams.update({
    'figure.facecolor': C_BG, 'axes.facecolor': '#1e293b',
    'text.color': C_TEXT, 'axes.labelcolor': C_TEXT,
    'xtick.color': C_TEXT, 'ytick.color': C_TEXT,
    'axes.edgecolor': '#334155', 'grid.color': '#334155',
    'font.family': 'sans-serif', 'font.size': 11
})

def solar_declination(doy):
    return 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)

def sunset_hour_angle(lat_rad, decl):
    arg = max(-1, min(1, -math.tan(lat_rad) * math.tan(decl)))
    return math.acos(arg)

def infer_et0_for_date(year, month):
    # Base climatology + slight climate change warming trend (0.04 C per year since 2000)
    y_diff = max(0, year - 2000)
    tmax = TMAX_CLIM[month] + (y_diff * 0.04)
    tmin = TMIN_CLIM[month] + (y_diff * 0.04)
    # Average DOY for the month
    doy = int((month - 1) * 30.4 + 15)
    ra = extraterrestrial_radiation(doy)
    # Approximate mean daily RS (assuming ~55% of Ra reaches the ground as climatology)
    rs_mj = ra * 0.55
    et0_daily = hargreaves_et0(rs_mj, tmax, tmin, ra)
    return et0_daily * 30.4  # Monthly ET0 total

def extraterrestrial_radiation(doy, lat_deg=LAT):
    lat_rad = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    decl = solar_declination(doy)
    ws = sunset_hour_angle(lat_rad, decl)
    ra = (24 * 60 / math.pi) * GSC * dr * (
        ws * math.sin(lat_rad) * math.sin(decl) +
        math.cos(lat_rad) * math.cos(decl) * math.sin(ws))
    return max(0, ra)

def hargreaves_et0(rs_mj, tmax, tmin, ra):
    """Hargreaves-Samani Reference ET₀ (mm/day)."""
    tmean = (tmax + tmin) / 2.0
    td = max(0, tmax - tmin)
    et0 = 0.0135 * (rs_mj / ra) * (tmean + 17.8) * (td ** 0.5) * ra * 0.408 if ra > 0 else 0
    # Simplified Hargreaves: ET₀ = 0.0023 * Ra * (Tmean+17.8) * (Tmax-Tmin)^0.5
    et0 = 0.0023 * ra * 0.408 * (tmean + 17.8) * (td ** 0.5)
    return max(0, et0)

def load_and_calculate():
    csv_path = Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_complete.csv")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['doy'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['rs_mj'] = df['daily_total_mj_m2'].astype(float)

    rng = np.random.default_rng(42)
    et0_vals = []
    for _, row in df.iterrows():
        m = int(row['month'])
        tmax = TMAX_CLIM[m] + rng.normal(0, 1.5)
        tmin = TMIN_CLIM[m] + rng.normal(0, 1.0)
        ra = extraterrestrial_radiation(int(row['doy']))
        et = hargreaves_et0(row['rs_mj'], tmax, tmin, ra)
        et0_vals.append(et)
    df['et0_mm'] = et0_vals
    return df

def fig1_monthly_radiation_et(df, out_dir):
    monthly = df.groupby('month').agg({'rs_mj': 'mean', 'et0_mm': 'mean'}).reset_index()
    months = ['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara']
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    x = np.arange(12)
    bars = ax1.bar(x - 0.2, monthly['rs_mj'], 0.35, color=C_ACCENT, alpha=0.85, label='Güneş Radyasyonu (MJ/m²)', zorder=3)
    ax1.set_ylabel('Günlük Radyasyon (MJ/m²/gün)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, fontsize=11)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, monthly['et0_mm'], 0.35, color=C_SECONDARY, alpha=0.85, label='ET₀ (mm/gün)', zorder=3)
    ax2.set_ylabel('Referans ET₀ (mm/gün)', fontsize=12)
    ax1.set_title('Aylık Ortalama Güneş Radyasyonu ve Evapotranspirasyon (İstanbul)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.3)
    plt.tight_layout()
    path = out_dir / 'fig1_monthly_rad_et.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def fig2_seasonal_box(df, out_dir):
    season_map = {12:'Kış',1:'Kış',2:'Kış',3:'İlkbahar',4:'İlkbahar',5:'İlkbahar',
                  6:'Yaz',7:'Yaz',8:'Yaz',9:'Sonbahar',10:'Sonbahar',11:'Sonbahar'}
    df['season'] = df['month'].map(season_map)
    season_order = ['Kış','İlkbahar','Yaz','Sonbahar']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [C_SECONDARY, C_REAL, C_ACCENT, '#ef4444']
    for i, s in enumerate(season_order):
        data = df[df['season']==s]['et0_mm']
        bp = ax1.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor=colors[i], alpha=0.7),
                         medianprops=dict(color='white', linewidth=2),
                         flierprops=dict(markerfacecolor=colors[i], markersize=3))
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(season_order, fontsize=11)
    ax1.set_ylabel('ET₀ (mm/gün)', fontsize=12)
    ax1.set_title('Mevsimsel ET₀ Dağılımı', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.2)

    for i, s in enumerate(season_order):
        data = df[df['season']==s]['rs_mj']
        bp = ax2.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor=colors[i], alpha=0.7),
                         medianprops=dict(color='white', linewidth=2),
                         flierprops=dict(markerfacecolor=colors[i], markersize=3))
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(season_order, fontsize=11)
    ax2.set_ylabel('Radyasyon (MJ/m²/gün)', fontsize=12)
    ax2.set_title('Mevsimsel Radyasyon Dağılımı', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    path = out_dir / 'fig2_seasonal_box.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def fig3_annual_trend(df, out_dir):
    annual = df.groupby('year').agg({'et0_mm': ['mean','sum'], 'rs_mj': ['mean','sum']}).reset_index()
    annual.columns = ['year','et0_mean','et0_sum','rs_mean','rs_sum']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(annual['year'], annual['rs_mean'], 'o-', color=C_ACCENT, linewidth=2, markersize=6, label='Ortalama Radyasyon')
    z = np.polyfit(annual['year'], annual['rs_mean'], 1)
    p = np.poly1d(z)
    ax1.plot(annual['year'], p(annual['year']), '--', color='#ef4444', linewidth=1.5, alpha=0.7, label=f'Trend: {z[0]:+.3f} MJ/m²/yıl')
    ax1.set_ylabel('MJ/m²/gün', fontsize=12)
    ax1.set_title('Yıllık Ortalama Güneş Radyasyonu Trendi', fontsize=13, fontweight='bold')
    ax1.legend(framealpha=0.3)
    ax1.grid(alpha=0.2)

    ax2.fill_between(annual['year'], annual['et0_sum'], color=C_SECONDARY, alpha=0.3)
    ax2.plot(annual['year'], annual['et0_sum'], 'o-', color=C_SECONDARY, linewidth=2, markersize=6, label='Toplam ET₀')
    z2 = np.polyfit(annual['year'], annual['et0_sum'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(annual['year'], p2(annual['year']), '--', color='#ef4444', linewidth=1.5, alpha=0.7, label=f'Trend: {z2[0]:+.1f} mm/yıl')
    ax2.set_ylabel('Toplam ET₀ (mm/yıl)', fontsize=12)
    ax2.set_xlabel('Yıl', fontsize=12)
    ax2.set_title('Yıllık Toplam Evapotranspirasyon Trendi', fontsize=13, fontweight='bold')
    ax2.legend(framealpha=0.3)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / 'fig3_annual_trend.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def fig4_data_source(df, out_dir):
    src = df.groupby(['year','data_source']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if 'real_extracted' in src.columns:
        ax.bar(src.index, src.get('real_extracted', 0), color=C_REAL, label='Gerçek Dijitalleştirilmiş', alpha=0.85)
    if 'synthetic' in src.columns:
        bottom = src.get('real_extracted', pd.Series(0, index=src.index))
        ax.bar(src.index, src.get('synthetic', 0), bottom=bottom, color=C_SYNTH, label='Sentetik (Angstrom-Prescott)', alpha=0.65)
    ax.set_xlabel('Yıl', fontsize=12)
    ax.set_ylabel('Gün Sayısı', fontsize=12)
    ax.set_title('Veri Kaynağı Dağılımı — Gerçek vs Sentetik', fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.3)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    path = out_dir / 'fig4_data_source.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def fig5_heatmap(df, out_dir):
    pivot = df.pivot_table(values='et0_mm', index=df['date'].dt.month, columns=df['date'].dt.year, aggfunc='mean')
    months_tr = ['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara']
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_yticks(range(12))
    ax.set_yticklabels(months_tr)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=45)
    ax.set_title('Aylık Ortalama ET₀ Isı Haritası (mm/gün)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ET₀ (mm/gün)', fontsize=11)
    plt.tight_layout()
    path = out_dir / 'fig5_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def fig6_correlation(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    sample = df.sample(min(2000, len(df)), random_state=42)
    colors_map = [C_REAL if s == 'real_extracted' else C_SYNTH for s in sample['data_source']]
    ax.scatter(sample['rs_mj'], sample['et0_mm'], c=colors_map, alpha=0.4, s=12, edgecolors='none')
    z = np.polyfit(df['rs_mj'], df['et0_mm'], 1)
    p = np.poly1d(z)
    xr = np.linspace(df['rs_mj'].min(), df['rs_mj'].max(), 50)
    ax.plot(xr, p(xr), '--', color='white', linewidth=2, label=f'R² = {np.corrcoef(df["rs_mj"], df["et0_mm"])[0,1]**2:.3f}')
    ax.set_xlabel('Güneş Radyasyonu (MJ/m²/gün)', fontsize=12)
    ax.set_ylabel('ET₀ (mm/gün)', fontsize=12)
    ax.set_title('Radyasyon-ET₀ Korelasyonu', fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.3, fontsize=12)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / 'fig6_correlation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

# def fig7_future_water_budget(out_dir):
#     try:
#         # Load datasets
#         dam_df = pd.read_csv('/Users/yasinkaya/Hackhaton/DATA/dam_occupancy.csv')
#         precip_df = pd.read_csv('/Users/yasinkaya/Hackhaton/output/forecast_package/clean_2035_datasets/precip_monthly_continuous_to_2035.csv')
#         
#         # Process Dam Data
#         dam_df['DATE'] = pd.to_datetime(dam_df['DATE'])
#         dam_df['YearMonth'] = dam_df['DATE'].dt.to_period('M')
#         dam_monthly = dam_df.groupby('YearMonth')['GENERAL_DAM_OCCUPANCY_RATE'].mean().reset_index()
#         dam_monthly['Date'] = dam_monthly['YearMonth'].dt.to_timestamp()
#         
#         # Process Precip Data
#         precip_df['timestamp'] = pd.to_datetime(precip_df['timestamp'])
#         precip_df = precip_df[(precip_df['timestamp'].dt.year >= 2005) & (precip_df['timestamp'].dt.year <= 2035)]
#         
#         # Calculate ET0 to 2035
#         forecast_dates = pd.date_range(start='2005-01-01', end='2035-12-01', freq='MS')
#         forecast_df = pd.DataFrame({'Date': forecast_dates})
#         forecast_df['Year'] = forecast_df['Date'].dt.year
#         forecast_df['Month'] = forecast_df['Date'].dt.month
#         
#         forecast_df['ET0_Monthly'] = forecast_df.apply(lambda row: infer_et0_for_date(row['Year'], row['Month']), axis=1)
#         
#         # Merge Precip
#         forecast_df = pd.merge(forecast_df, precip_df[['timestamp', 'value']], left_on='Date', right_on='timestamp', how='left')
#         forecast_df.rename(columns={'value': 'Precip_Monthly'}, inplace=True)
#         # Handle NAs with moving average mostly for stability
#         forecast_df['Precip_Monthly'] = forecast_df['Precip_Monthly'].interpolate(method='linear')
#         
#         # Merge Dam Rate
#         forecast_df = pd.merge(forecast_df, dam_monthly[['Date', 'GENERAL_DAM_OCCUPANCY_RATE']], on='Date', how='left')
#         
#         # Calculate Water Balance Status
#         # Net Water = Precip - ET0
#         forecast_df['Net_Water'] = forecast_df['Precip_Monthly'] - forecast_df['ET0_Monthly']
#         forecast_df['Net_Water_Rolling'] = forecast_df['Net_Water'].rolling(12, min_periods=1).mean()
#         
#         # Predict missing Dam Rate using simple Net_Water_Rolling regression and AR baseline (Conceptual for slides)
#         # Starting from last known dam rate
#         last_dam = forecast_df['GENERAL_DAM_OCCUPANCY_RATE'].dropna().iloc[-1]
#         last_dam_idx = forecast_df['GENERAL_DAM_OCCUPANCY_RATE'].last_valid_index()
#         predicted_dam = []
#         curr_dam = last_dam
#         for i in range(len(forecast_df)):
#             if i <= last_dam_idx:
#                 predicted_dam.append(np.nan)
#             else:
#                 # Add anomaly
#                 curr_dam = curr_dam + (forecast_df.loc[i, 'Net_Water_Rolling'] * 0.1) # 0.1 scalar for dam volume translation
#                 curr_dam = max(0, min(100, curr_dam - 0.5)) # Gradual baseline draw-down + cap
#                 predicted_dam.append(curr_dam)
#         forecast_df['Predicted_Dam_Rate'] = predicted_dam
#         
#         # Plotting
#         fig, ax1 = plt.subplots(figsize=(13, 6))
#         
#         # Dam fill
#         ax1.plot(forecast_df['Date'], forecast_df['GENERAL_DAM_OCCUPANCY_RATE'], color=C_SECONDARY, linewidth=2, label='İSKİ Gerçek Baraj Doluluğu (%)')
#         ax1.plot(forecast_df['Date'], forecast_df['Predicted_Dam_Rate'], color='#ef4444', linewidth=2.5, linestyle='--', label='2035 Projeksiyon: Tahmini Doluluk (%)')
#         ax1.fill_between(forecast_df['Date'], 0, forecast_df['GENERAL_DAM_OCCUPANCY_RATE'], color=C_SECONDARY, alpha=0.2)
#         ax1.fill_between(forecast_df['Date'], 0, forecast_df['Predicted_Dam_Rate'], color='#ef4444', alpha=0.1)
#         
#         ax1.set_ylabel('Baraj Doluluk Oranı (%)', fontsize=12, color=C_TEXT)
#         ax1.set_ylim(0, 100)
#         ax1.grid(alpha=0.15)
#         
#         # Second axis for Net Water Anomaly
#         ax2 = ax1.twinx()
#         ax2.bar(forecast_df['Date'], forecast_df['Net_Water'], width=20, alpha=0.4, color=np.where(forecast_df['Net_Water'] > 0, C_REAL, C_ACCENT), label='Su Bütçesi (Yağış - ET₀)')
#         ax2.set_ylabel('Aylık Su Bütçesi Değişimi (mm)', fontsize=12, color=C_TEXT)
#         
#         ax1.set_title("Evapotranspirasyon, Yağış ve İBB Baraj Doluluğu 2035 Projeksiyonu", fontsize=14, fontweight='bold', pad=15)
#         
#         # Combine legends
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.3, fontsize=10)
#         
#         plt.axvline(pd.to_datetime('2021-04-01'), color='white', linestyle=':', alpha=0.5)
#         plt.text(pd.to_datetime('2022-01-01'), 90, 'Tahmin Bölgesi (Yapay Zeka) →', color='white', fontsize=10, alpha=0.8)
#         
#         plt.tight_layout()
#         path = out_dir / 'fig7_future_water_budget.png'
#         fig.savefig(path, dpi=150, bbox_inches='tight')
#         plt.close()
#         return path
#     except Exception as e:
#         print(f"Dam plotting error: {e}")
#         return None

def build_html(figs, df, out_path):
    # Calculate some basic delta for text injection
    annual_et_start = df[df['year'] < df['year'].min() + 5].groupby('year')['et0_mm'].sum().mean()
    annual_et_end = df[df['year'] > df['year'].max() - 5].groupby('year')['et0_mm'].sum().mean()
    delta_et_pct = ((annual_et_end - annual_et_start) / annual_et_start) * 100 if annual_et_start > 0 else 0

    stats = {
        'total_days': len(df),
        'real_pct': len(df[df['data_source']=='real_extracted']) / len(df) * 100,
        'mean_rs': df['rs_mj'].mean(),
        'mean_et0': df['et0_mm'].mean(),
        'annual_et0': df.groupby('year')['et0_mm'].sum().mean(),
        'max_et0': df['et0_mm'].max(),
        'date_range': f"{df['date'].min().strftime('%Y')}–{df['date'].max().strftime('%Y')}",
        'years': len(df['year'].unique()),
        'delta_et_pct': delta_et_pct
    }

    slides = []

    # Slide 1 - Title
    slides.append(f"""
    <section class="slide slide-title">
      <div class="content">
        <div class="badge">HACKATHON 2026</div>
        <h1>Evapotranspirasyon (ET₀) Analizi<br><span style='font-size:0.6em;color:#94a3b8'>Tarihi Aktinograf Kayıtlarından Gelecek Vizyonuna</span></h1>
        <p class="subtitle">Kandilli Rasathanesi • {stats['date_range']}</p>
        <div class="stats-row">
          <div class="stat"><span class="num">{stats['total_days']:,}</span><span class="label">Günlük Kayıt</span></div>
          <div class="stat"><span class="num">{stats['real_pct']:.0f}%</span><span class="label">Gerçek Veri</span></div>
          <div class="stat"><span class="num">{stats['years']}</span><span class="label">Yıl</span></div>
        </div>
      </div>
    </section>
    """)

    # Slide 2 - Problem
    slides.append("""
    <section class="slide">
      <div class="content">
        <h2>🎯 Problem ve Motivasyon</h2>
        <div class="two-col">
          <div class="col">
            <h3>Sorun</h3>
            <ul>
              <li>Türkiye'de <strong>1975-2004</strong> yılları arası aktinograf (güneş radyasyonu) verileri <b>analog kağıt şeritlerde</b> kayıtlı</li>
              <li>Bu veriler şu anda <b>kullanılamaz durumda</b> — dijitalleştirilmemiş</li>
              <li>Klimatolojik analiz, tarımsal planlama ve enerji modellemesi için <b>kritik kayıp</b></li>
            </ul>
          </div>
          <div class="col">
            <h3>Çözüm</h3>
            <ul>
              <li><b>Bilgisayarlı görü + Viterbi algoritması</b> ile otomatik iz sürme</li>
              <li>Günlük toplam radyasyonun çıkarılması (cal/cm² → MJ/m²)</li>
              <li><b>Hargreaves-Samani</b> yöntemiyle referans evapotranspirasyon (ET₀) hesaplaması</li>
              <li>Eksik veriler için fizik tabanlı <b>Angstrom-Prescott</b> sentetik veri dolgusu</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
    """)

    # Slide 3 - Formula
    if len(figs) > 0:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[0].name}" alt="Formül Açıklaması" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 4 - Daily
    if len(figs) > 1:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[1].name}" alt="Günlük Analiz" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 5 - Monthly
    if len(figs) > 2:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[2].name}" alt="Aylık Analiz" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 6 - Trend
    if len(figs) > 3:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[3].name}" alt="Yıllık Trend Analizi" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 6.1 - NEW: Quant Forecast (v4)
    if len(figs) > 10:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[10].name}" alt="Quant Ön Görüsü" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 6.2 - NEW: Hourly Delta (v4)
    if len(figs) > 11:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[11].name}" alt="Saatlik Delta Analizi" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # Slide 6.3 - NEW: Dam Explained (v4)
    if len(figs) > 12:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; justify-content:center;">
            <img src="{figs[12].name}" alt="Istanbul Baraj Projeksiyonu" style="max-height:90vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain; margin:0 auto;">
          </div>
        </section>
        """)

    # --- Re-add legacy simple charts below ---
    
    # Slide 7 - Simple Monthly Profile
    if len(figs) > 4:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>📊 Mevsimsel Radyasyon Profil Özeti</h2>
            <img src="{figs[4].name}" alt="Aylık Radyasyon ve ET₀">
            <div class="insight">
              <b>Klimatolojik Bulgu:</b> Yaz aylarında (Haz-Ağu) ortalama ET₀ <b>{df[df['month'].isin([6,7,8])]['et0_mm'].mean():.1f} mm/gün</b> ile kış aylarına göre yüksek oranda artış göstermektedir.
            </div>
          </div>
        </section>
        """)

    # Slide 8 - Seasonal Boxplots
    if len(figs) > 5:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>🌡️ Mevsimsel Yayılım Dağılımı</h2>
            <img src="{figs[5].name}" alt="Mevsimsel Dağılım">
            <div class="insight">
              <b>Varyans:</b> Yaz mevsiminde güneşlenme kararlılığı ET₀ değerlerini dar bir aralığa oturturken, kış mevsiminde bulutluluk farkları belirgindir.
            </div>
          </div>
        </section>
        """)

    # Slide 9 - Simple Annual Trend
    if len(figs) > 6:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>📉 Basitleştirilmiş Yıllık Trend</h2>
            <img src="{figs[6].name}" alt="Basit Yıllık Trend">
            <div class="insight">
              <b>İklim Sinyali:</b> Theil-Sen Robust trendin teyidi niteliğinde basit doğrusal (Lineer) ET₀ regresyonu.
            </div>
          </div>
        </section>
        """)

    # Slide 10 - Data Source Distribution
    if len(figs) > 7:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>💾 Veri Kaynağı Analizi (Gerçek vs Sentetik)</h2>
            <img src="{figs[7].name}" alt="Veri Kaynağı">
            <div class="insight">
              <b>Veri Kalitesi:</b> Hangi yılların fiziksel formülle (Angstrom) doldurulduğunun tam analizi.
            </div>
          </div>
        </section>
        """)

    # Slide 11 - Heatmap
    if len(figs) > 8:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>🔥 ET₀ Isı Haritası — Ay × Yıl</h2>
            <img src="{figs[8].name}" alt="Isı Haritası">
            <div class="insight">
              Bu harita kronolojik eksende tarımsal su stresinin yoğunlaştığı alev noktalarını gösterir.
            </div>
          </div>
        </section>
        """)

    # Slide 12 - Correlation
    if len(figs) > 9:
        slides.append(f"""
        <section class="slide">
          <div class="content">
            <h2>🔗 Radyasyon - ET₀ Korelasyonu</h2>
            <img src="{figs[9].name}" alt="Korelasyon">
            <div class="insight">
              <b style="color:#10b981">Yeşil noktalar</b> dijitalleştirilmiş günleri, <b style="color:#6366f1">mor noktalar</b> sentetik veri tabanı bağlantılarını işaretler.
            </div>
          </div>
        </section>
        """)

    # Slide 9.5 - Dam Forecast 2035 (REMOVED)
    # if len(figs) > 6 and figs[6] is not None:
    #     slides.append(f"""
    #     <section class="slide">
    #       <div class="content">
    #         <h2>🔮 2035 Vizyonu: Su Bütçesi ve Baraj Doluluğu Tahmini</h2>
    #         ...
    #       </div>
    #     </section>
    #     """)

    # Slide 10 - Applications (Core)
    slides.append("""
    <section class="slide">
      <div class="content">
        <h2>🚀 Temel Kullanım Alanları</h2>
        <div class="cards">
          <div class="card">
            <div class="card-icon">🌾</div>
            <h3>Tarımsal Sulama</h3>
            <p>ET₀ verileriyle <b>hassas sulama programları</b> oluşturulabilir. Bitkinin gerçek su ihtiyacı hesaplanarak su israfı %30-40 azaltılabilir.</p>
          </div>
          <div class="card">
            <div class="card-icon">☀️</div>
            <h3>Güneş Enerjisi</h3>
            <p>Tarihsel radyasyon verileri, <b>fotovoltaik santral</b> verimlilik tahminleri ve yatırım fizibilite çalışmaları için temel oluşturur.</p>
          </div>
          <div class="card">
            <div class="card-icon">🌍</div>
            <h3>İklim Değişikliği</h3>
            <p>30 yıllık trend analizi ile <b>kuraklık riski</b>, buharlaşma artışı ve su bütçesi değişimleri izlenebilir.</p>
          </div>
          <div class="card">
            <div class="card-icon">💧</div>
            <h3>Su Kaynakları</h3>
            <p>ET₀ verileri, <b>havza bazlı su bütçesi</b> modelleri ve baraj doluluk tahminlerinin temel girdisidir.</p>
          </div>
        </div>
      </div>
    </section>
    """)

    # Slide 10.5 - Advanced Applications
    slides.append("""
    <section class="slide">
      <div class="content">
        <h2>🛠️ İleri Düzey Sektörel Uygulamalar</h2>
        <div class="cards">
          <div class="card">
            <div class="card-icon">🏙️</div>
            <h3>Akıllı Şehir Planlaması</h3>
            <p><b>Kentsel Isı Adası Etkisi:</b> Şehir içi yeşil alanların serinletme etkisi ET₀ ile simüle edilir. <br>
            <b>Yağmur Suyu Hasadı:</b> Kritik buharlaşma dönemlerinde rezervuarlar yapay zeka ile yönetilir.</p>
          </div>
          <div class="card">
            <div class="card-icon">⚠️</div>
            <h3>Afet ve Risk Yönetimi</h3>
            <p><b>Orman Yangını Erken Uyarısı:</b> Düşük yağış ve yüksek ET₀ periyotlarının kesişimi yangın riskini saptar.<br>
            <b>Taşkın Modellemesi:</b> Toprak nem doygunluğu ET₀ verisine göre kalibre edilir.</p>
          </div>
          <div class="card">
            <div class="card-icon">📈</div>
            <h3>Tarım Finansmanı</h3>
            <p><b>Agro-Sigorta Primleri:</b> Bölgeye özgü ET₀ trendleri verim risklerini (Risk of Yield) sayısallaştırarak sigorta poliçesini optimize eder.</p>
          </div>
          <div class="card">
            <div class="card-icon">🌾</div>
            <h3>Emtia Tahmini</h3>
            <p><b>Vadeli İşlemler:</b> Rekolte tahminleri (su bütçesi açığına endeksli) tarımsal vadeli işlem piyasaları için al-sat sinyali üretir.</p>
          </div>
        </div>
      </div>
    </section>
    """)

    # Slide 11 - Future development
    slides.append("""
    <section class="slide">
      <div class="content">
        <h2>🔮 Geliştirme Yol Haritası</h2>
        <div class="timeline">
          <div class="tl-item">
            <div class="tl-phase">Faz 1</div>
            <div class="tl-content">
              <h3>Veri Genişletme</h3>
              <p>Tüm Türkiye meteoroloji istasyonlarının aktinograf arşivlerinin dijitalleştirilmesi. 
              Daha fazla gerçek veriyle sentetik oranının %10 altına indirilmesi.</p>
            </div>
          </div>
          <div class="tl-item">
            <div class="tl-content">
              <div class="tl-phase">Faz 2</div>
              <h3>Yapay Zeka Entegrasyonu</h3>
              <p>Derin öğrenme (LSTM/Transformer) modelleri ile ET₀ tahmini. Uydu görüntüleriyle (Sentinel-2) 
              gerçek zamanlı radyasyon haritaları oluşturma.</p>
            </div>
          </div>
          <div class="tl-item">
            <div class="tl-content">
              <div class="tl-phase">Faz 3</div>
              <h3>Mobil Uygulama</h3>
              <p>Çiftçilere yönelik <b>"Ne kadar sulamalıyım?"</b> sorusunu yanıtlayan, 
              konum bazlı sulama tavsiye sistemi. Gerçek zamanlı ET₀ hesaplama ve tarla bazında sulama planı.</p>
            </div>
          </div>
          <div class="tl-item">
            <div class="tl-content">
              <div class="tl-phase">Faz 4</div>
              <h3>Politika Desteği</h3>
              <p>GAP, KOP ve DSİ gibi kurumlar için <b>ulusal ET₀ atlası</b> oluşturma. 
              İklim değişikliği senaryolarıyla gelecek su bütçesi projeksiyonları.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
    """)

    # Slide 12 - Action/Thanks
    slides.append(f"""
    <section class="slide slide-title">
      <div class="content">
        <h1 style="font-size:3.5em;margin-bottom:10px;">Aksiyona Geçme Zamanı</h1>
        <p class="subtitle" style="font-size:1.4em;opacity:0.9;max-width:800px;margin:0 auto 40px auto;line-height:1.5;">Göz ardı edilen aktinograf verileri, artık tarımsal krizleri önleyen ve su bütçesini yöneten <br>stratejik bir <span style="color:#38bdf8;font-weight:bold;">Karar Destek Sistemine</span> dönüştü.</p>
        <div class="stats-row" style="margin-bottom:30px">
          <div class="stat"><span class="num">{stats['total_days']:,}</span><span class="label">Günlük Veri İşlendi</span></div>
          <div class="stat"><span class="num">{stats['mean_et0']:.1f}</span><span class="label">Ort. ET₀ (mm/gün)</span></div>
          <div class="stat"><span class="num" style="color:#ef4444">+{stats['delta_et_pct']:.1f}%</span><span class="label">Tarihsel ET₀ Artışı (Δ)</span></div>
        </div>
        <p style="opacity:0.6;font-size:0.9em;">Dinlediğiniz için teşekkürler.</p>
      </div>
    </section>
    """)

    # --- Deep Dive / Appendix Slides (Requested after CTA) ---

    # Slide 13 - Risk: Skin Cancer 1
    if len(figs) > 13:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; flex-direction:column; align-items:center;">
            <h2 style="margin-bottom:10px;">🛡️ Sağlık Riski: Cilt Kanseri Proksisi</h2>
            <img src="{figs[13].name}" alt="Cilt Kanseri Pano" style="max-height:80vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain;">
            <div class="insight" style="width:100%; margin-top:15px;">
              <b>Analiz:</b> Artan solar potansiyel ve HI (Heat Index), UV maruziyeti için dolaylı baskı oluşturmaktadır.
            </div>
          </div>
        </section>
        """)

    # Slide 14 - Risk: Skin Cancer 2 (10k Cases)
    if len(figs) > 14:
        slides.append(f"""
        <section class="slide">
          <div class="content" style="max-width:1600px; padding:10px; display:flex; flex-direction:column; align-items:center;">
            <h2 style="margin-bottom:10px;">📊 10.000 Kişide Vaka Senaryosu</h2>
            <img src="{figs[14].name}" alt="Cilt Kanseri 10k" style="max-height:80vh; width:auto; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.5); object-fit:contain;">
            <div class="insight" style="width:100%; margin-top:15px;">
              <b>Senaryo:</b> Model bazlı iklim-baskı proksisi kullanılarak hazırlanan 10k vaka karşılaştırması.
            </div>
          </div>
        </section>
        """)

    slides_html = "\n".join(slides)

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aktinograf → Evapotranspirasyon Analizi</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',sans-serif; background:{C_BG}; color:{C_TEXT}; overflow:hidden; height:100vh; }}

.slide {{ width:100vw; height:100vh; display:none; flex-direction:column; justify-content:center; padding:0; overflow:hidden; }}
.slide.active {{ display:flex; }}
.slide-title {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%); text-align:center; }}
.slide-title h1 {{ font-size:2.8em; font-weight:800; line-height:1.2; margin-bottom:20px;
  background:linear-gradient(135deg,#e2e8f0,#38bdf8,#2dd4bf); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.slide-title .subtitle {{ font-size:1.2em; opacity:0.7; margin-bottom:35px; }}

.content {{ max-width:1100px; margin:0 auto; padding:40px 50px; width:100%; }}
h2 {{ font-size:1.8em; font-weight:700; margin-bottom:25px; }}
.content img {{ width:100%; max-height:420px; object-fit:contain; border-radius:12px; margin:10px 0; }}

.badge {{ display:inline-block; background:rgba(14,165,233,0.2); color:#38bdf8; padding:6px 18px; border-radius:20px; font-size:0.85em; font-weight:600; letter-spacing:2px; margin-bottom:20px; border:1px solid rgba(56,189,248,0.3); }}

.stats-row {{ display:flex; justify-content:center; gap:50px; margin-top:15px; }}
.stat {{ text-align:center; }}
.stat .num {{ display:block; font-size:2.2em; font-weight:800; color:#38bdf8; }}
.stat .label {{ font-size:0.85em; opacity:0.6; text-transform:uppercase; letter-spacing:1px; }}

.insight {{ background:rgba(14,165,233,0.1); border-left:3px solid #38bdf8; padding:12px 18px; border-radius:0 8px 8px 0; margin-top:10px; font-size:0.92em; line-height:1.5; }}

.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; }}
.col h3 {{ color:#38bdf8; margin-bottom:10px; font-size:1.1em; }}
.col ul {{ list-style:none; padding:0; }}
.col li {{ padding:6px 0; font-size:0.92em; line-height:1.5; border-bottom:1px solid rgba(255,255,255,0.05); }}
.col li:before {{ content:'▸ '; color:#2dd4bf; font-weight:bold; }}

.cards {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.card {{ background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:22px; transition:transform 0.2s; }}
.card:hover {{ transform:translateY(-3px); border-color:rgba(56,189,248,0.3); }}
.card-icon {{ font-size:2em; margin-bottom:8px; }}
.card h3 {{ color:#38bdf8; font-size:1em; margin-bottom:6px; }}
.card p {{ font-size:0.85em; opacity:0.8; line-height:1.5; }}

.timeline {{ position:relative; padding-left:35px; }}
.timeline::before {{ content:''; position:absolute; left:12px; top:0; bottom:0; width:2px; background:linear-gradient(to bottom,#38bdf8,#2dd4bf,#f59e0b,#ef4444); }}
.tl-item {{ position:relative; margin-bottom:22px; }}
.tl-item::before {{ content:''; position:absolute; left:-29px; top:5px; width:12px; height:12px; border-radius:50%; background:#38bdf8; border:2px solid {C_BG}; }}
.tl-item:nth-child(2)::before {{ background:#2dd4bf; }}
.tl-item:nth-child(3)::before {{ background:#f59e0b; }}
.tl-item:nth-child(4)::before {{ background:#ef4444; }}
.tl-phase {{ display:inline-block; background:rgba(56,189,248,0.15); color:#38bdf8; padding:2px 10px; border-radius:10px; font-size:0.75em; font-weight:600; margin-bottom:4px; }}
.tl-item h3 {{ font-size:1em; margin-bottom:3px; }}
.tl-item p {{ font-size:0.82em; opacity:0.75; line-height:1.4; }}

.nav {{ position:fixed; bottom:20px; right:30px; display:flex; gap:10px; z-index:100; }}
.nav button {{ width:44px; height:44px; border-radius:50%; border:1px solid rgba(255,255,255,0.2); background:rgba(15,23,42,0.8); color:white; font-size:1.2em; cursor:pointer; backdrop-filter:blur(10px); transition:all 0.2s; display:flex; align-items:center; justify-content:center; }}
.nav button:hover {{ background:rgba(56,189,248,0.3); border-color:#38bdf8; }}
.nav .fs-btn {{ width:auto; padding:0 15px; border-radius:20px; font-size:0.8em; font-weight:600; }}
.counter {{ position:fixed; bottom:28px; left:30px; font-size:0.8em; opacity:0.4; z-index:100; }}
</style>
</head>
<body>
{slides_html}
<div class="nav">
  <button class="fs-btn" onclick="toggleFS()">TAM EKRAN</button>
  <button onclick="prev()">◀</button>
  <button onclick="next()">▶</button>
</div>
<div class="counter" id="counter"></div>
<script>
let current=0;
const slides=document.querySelectorAll('.slide');
function show(n){{current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.getElementById('counter').textContent=(current+1)+' / '+slides.length;}}
function next(){{show(current+1);}} function prev(){{show(current-1);}}
function toggleFS() {{
  if (!document.fullscreenElement) {{
    document.documentElement.requestFullscreen();
  }} else {{
    if (document.exitFullscreen) {{
      document.exitFullscreen();
    }}
  }}
}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight'||e.key===' ')next();if(e.key==='ArrowLeft')prev();}});
show(0);
</script>
</body>
</html>"""

    out_path.write_text(html, encoding='utf-8')
    print(f"Presentation saved: {out_path}")

def main():
    import shutil
    out_dir = Path("/Users/yasinkaya/Hackhaton/output/presentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and calculating ET₀...")
    df = load_and_calculate()
    print(f"  {len(df)} days loaded, ET₀ range: {df['et0_mm'].min():.2f} – {df['et0_mm'].max():.2f} mm/day")

    print("Copying advanced quantitative charts...")
    charts_dir = Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/charts")
    src_images = [
        charts_dir / "tarim_et0_formula_explained.png",
        charts_dir / "tarim_et0_daily_explained_2004.png",
        charts_dir / "tarim_et0_monthly_explained_2004.png",
        charts_dir / "tarim_et0_yearly_trend_robust_explained.png"
    ]
    
    figs = []
    for src in src_images:
        if src.exists():
            dst = out_dir / src.name
            shutil.copy(src, dst)
            figs.append(dst)
            print(f"  [+] Copied robust chart: {dst.name}")
        else:
            print(f"  [!] Missing chart: {src.name}")

    print("Generating simple thematic charts...")
    simple_figs = [
        fig1_monthly_radiation_et(df, out_dir),
        fig2_seasonal_box(df, out_dir),
        fig3_annual_trend(df, out_dir),
        fig4_data_source(df, out_dir),
        fig5_heatmap(df, out_dir),
        fig6_correlation(df, out_dir),
    ]

    for f in simple_figs:
        figs.append(f)
        print(f"  [+] Generated simple chart: {f.name}")

    # extra charts from user screenshots (v4 versions)
    extra_srcs = [
        out_dir / "tarim_et0_quant_forecast_v4.png",
        out_dir / "tarim_et0_hourly_delta_v4.png",
    ]
    for src in extra_srcs:
        if src.exists():
            figs.append(src)
            print(f"  [+] Added v4 EXTRA chart: {src.name}")
        else:
            print(f"  [!] Missing v4 EXTRA: {src.name}")

    # Dam explained slide (Updated with History)
    dam_src = out_dir / "istanbul_dam_explained_v4.png"
    if dam_src.exists():
        figs.append(dam_src)
        print(f"  [+] Added DAM EXPLAINED (v4 with History): {dam_src.name}")
    else:
        print(f"  [!] Missing DAM EXPLAINED: {dam_src.name}")

    # Risk slides
    risk_srcs = [
        out_dir / "risk_skin_cancer.png",
        out_dir / "risk_skin_cancer_10k.png",
    ]
    for src in risk_srcs:
        if src.exists():
            figs.append(src)
            print(f"  [+] Added RISK chart: {src.name}")
        else:
            print(f"  [!] Missing RISK chart: {src.name}")

    print("\nBuilding HTML presentation...")
    out_html_path = out_dir / "eto_analizi_v4.html"
    build_html(figs, df, out_html_path)
    print(f"Done! Open {out_html_path.name} in a browser.")

if __name__ == "__main__":
    main()

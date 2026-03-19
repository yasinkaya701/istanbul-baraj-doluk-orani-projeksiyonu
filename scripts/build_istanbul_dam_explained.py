#!/usr/bin/env python3
"""Premium Explained Slide for Istanbul Dam Projections."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import et0_visual_style as style

def build_dam_explained_slide(csv_path: Path, out_path: Path):
    colors = style.theme()
    
    # Load data
    df = pd.read_csv(csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Filter for overall_mean as it represents the city-wide projection
    subset = df[df['series'] == 'overall_mean'].copy()
    
    fig = plt.figure(figsize=(16, 9), facecolor=colors["fig_bg"])
    
    # Grid layout: Left (Main Chart), Right (Explanation Sidebar)
    # Ratio 7:3
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 3], wspace=0.15, left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    ax_main = fig.add_subplot(gs[0])
    ax_sidebar = fig.add_subplot(gs[1])
    
    # --- Main Chart ---
    style.style_axes(ax_main, colors)
    
    # First, plot historical data for context (last 5-10 years)
    hist_df = df[df['series'] == 'overall_mean'].dropna(subset=['observed']).sort_values('ds')
    if not hist_df.empty:
        # Filter to show from ~2020 onwards for clarity in context
        ax_main.plot(hist_df['ds'], hist_df['observed'] * 100, 
                    color='#1e293b', linewidth=2, label='Gözlem (Tarihsel)', alpha=0.8)

    # Plot Scenarios
    scenarios = {
        'baseline': ('#225860', '-', 'Temel Senaryo (Beklenen)'),
        'dry_severe': ('#a54934', '--', 'Siddetli Kuraklik (Risk)'),
        'wet_severe': ('#b79a78', ':', 'Iyimser (Yagisli)')
    }
    
    first_ds = None
    for scen_id, (color, ls, label) in scenarios.items():
        # Only plot future part for scenarios
        scen_df = subset[(subset['scenario'] == scen_id) & (subset['ds'] >= '2025-01-01')].sort_values('ds')
        if scen_df.empty: continue
        
        if first_ds is None:
            first_ds = scen_df['ds'].iloc[0]

        ax_main.plot(scen_df['ds'], scen_df['scenario_yhat'] * 100, 
                    color=color, linestyle=ls, linewidth=2.5, label=label)
        
        # Fill uncertainty for baseline
        if scen_id == 'baseline':
            ax_main.fill_between(scen_df['ds'], 
                               scen_df['scenario_yhat_lower'] * 100, 
                               scen_df['scenario_yhat_upper'] * 100, 
                               color=color, alpha=0.1)

    # Thresholds
    ax_main.axhline(40, color='#685c52', linestyle='--', alpha=0.5, linewidth=1)
    if first_ds:
        ax_main.text(first_ds, 41, "Kritik Eşik (%40)", color='#685c52', fontsize=10, fontweight='bold')
    
    ax_main.set_title("İstanbul Baraj Doluluk Projeksiyonu (2025-2027)", 
                     fontsize=18, fontweight='bold', pad=20, color=colors["text"])
    ax_main.set_ylabel("Doluluk Oranı (%)", fontsize=12, labelpad=10)
    ax_main.set_ylim(0, 105)
    ax_main.legend(loc='upper right', frameon=False)
    
    # --- Sidebar ---
    style.setup_panel_axis(ax_sidebar, colors)
    
    # Context Card
    style.add_card(ax_sidebar, 0.05, 0.78, 0.9, 0.18, 
                  "Stratejik Özet", 
                  "İstanbul su rezervleri, artan buharlaşma\n"
                  "ve düzensiz yağış rejimi nedeniyle\n"
                  "yüksek iklim riskine maruzdur.\n"
                  "Projeksiyonlar 2026 yazını kritik buluyor.",
                  colors=colors, facecolor=colors["card_alt"])
    
    # Forecast Details
    # Get max prob below 40
    max_risk = subset['scenario_prob_below_40'].max() * 100
    style.add_card(ax_sidebar, 0.05, 0.52, 0.9, 0.22, 
                  "Tahmin Detayları", 
                  f"- Model: Ensemble SARIMA/Prophet\n"
                  f"- Veri: 2004-2024 Arşivi\n"
                  f"- Max Risk: %{max_risk:.1f} (Eşik Altı)\n"
                  "- ET0 Etkisi: %15-20 (Yaz Kaybı)",
                  colors=colors, facecolor=colors["card_blue"])
    
    # Action Card
    style.add_card(ax_sidebar, 0.05, 0.22, 0.9, 0.26, 
                  "Yönetim Önerileri", 
                  "- Dinamik tarife modellemesi\n"
                  "- Buharlaşma önleyici bariyerler\n"
                  "- Melen III hattı önceliklendirme\n"
                  "- Tarımsal sulama kısıt yönetimi",
                  colors=colors, facecolor=colors["card_gold"])
    
    # Footer info
    style.add_card(ax_sidebar, 0.05, 0.02, 0.9, 0.16, 
                  "Veri Kaynağı", 
                  "İSKİ Baraj Verileri &\n"
                  "Kandilli Meteoroloji Arşivi\n"
                  "Analiz: AgroGuard Hub",
                  colors=colors, facecolor=colors["card_red"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    csv_path = Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/scenario_forecasts.csv")
    out_path = Path("/Users/yasinkaya/Hackhaton/output/presentation/istanbul_dam_explained_v4.png")
    build_dam_explained_slide(csv_path, out_path)
    print(f"Generated dam explained slide: {out_path}")

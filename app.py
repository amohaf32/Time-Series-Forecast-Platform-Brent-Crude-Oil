import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import build_master_dataset, engineer_features, process_uploaded_file
from models_engine import (
    train_and_evaluate_all, 
    calculate_risk_and_signals, 
    simulate_scenario, 
    calculate_feature_contributions
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TSFP - Brent Crude Oil Intelligence Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
    }
    
    .signal-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 18px;
        border-left: 6px solid #38bdf8;
        margin-bottom: 15px;
        height: 100%;
    }

    .section-header {
        color: #f1f5f9;
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    .calc-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* High-Contrast Legibility Fix for Dark Mode DataFrames */
    div[data-testid="stDataFrame"] table {
        color: #ffffff !important;
    }
    div[data-testid="stDataFrame"] td {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stDataFrame"] th {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        background-color: #1e293b !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA INITIALIZATION & CACHING ---
@st.cache_data(ttl=3600)
def load_all_data(force_refresh=False):
    df_raw = build_master_dataset(force_refresh=force_refresh)
    df_engineered = engineer_features(df_raw)
    return df_raw, df_engineered

@st.cache_data(ttl=3600)
def run_model_pipeline(_df_engineered):
    return train_and_evaluate_all(
        _df_engineered, 
        train_pct=0.8,
        lasso_alpha=0.01,
        rf_estimators=100,
        rf_depth=4,
        xgb_lr=0.1,
        xgb_depth=4,
        champion_selection_mode="Composite Rank (Recommended)"
    )


# --- MAIN APPLICATION ENTRYPOINT ---
def main():
    # Sidebar Setup
    st.sidebar.title("🛢️ TSFP Controls")
    
    # 1. Audience Mode Selector
    st.sidebar.subheader("🎯 Audience View Profile")
    layman_label = "📋 Quick Overview"
    prof_label = "🔬 Quantitative Analytics"
    
    audience_mode = st.sidebar.radio(
        "Select Target Audience View:",
        [layman_label, prof_label],
        # help="Switch between executive layman signals and professional quantitative analytics."
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Live Market Data"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Re-fetching latest market prices...")

    # Load Data & Models
    df_raw, df_engineered = load_all_data()
    model_results = run_model_pipeline(df_engineered)
    
    preds_df = model_results['predictions']
    metrics_df = model_results['metrics']
    multi_horizon_preds = model_results['multi_horizon_preds']
    champion_col = model_results.get('champion_col', 'XGBoost')
    
    # Compute Risk & Signal Analytics
    risk_summary = calculate_risk_and_signals(df_raw, preds_df, metrics_df, multi_horizon_preds, champion_col=champion_col)

    # Header Box
    st.markdown("""
    <div class="header-box">
        <div class="header-title">Brent Crude Oil Time Series Forecasting Platform (TSFP)</div>
        <div class="header-subtitle">Interactive Financial Analytics, Multi-Model Forecasting & Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    # Top Metric Bar
    st.markdown('<div class="section-header">Live Market Prices</div>', unsafe_allow_html=True)
    latest = df_raw.iloc[-1]
    prev = df_raw.iloc[-2]
    brent_change = latest['Brent_Crude'] - prev['Brent_Crude']
    brent_pct = (brent_change / prev['Brent_Crude']) * 100
    spread = latest['Brent_Crude'] - latest['WTI_Crude']

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Brent Spot Price", f"${latest['Brent_Crude']:.2f}", f"{brent_change:+.2f} ({brent_pct:+.2f}%)")
    col2.metric("WTI Spot Price", f"${latest['WTI_Crude']:.2f}")
    col3.metric("Brent-WTI Spread", f"${spread:.2f}")
    col4.metric("USD Index (DXY)", f"{latest['US_Dollar_Index']:.2f}")
    col5.metric("VIX Volatility Index", f"{latest['VIX_Index']:.2f}")

    st.markdown("---")

    # =========================================================================
    # VIEW MODE 1: LAYMAN / QUICK OVERVIEW (NON-PROFESSIONAL AUDIENCE)
    # =========================================================================
    if audience_mode == layman_label:
        st.markdown('<div class="section-header">🔰 Executive Trade Summary & Market Signal</div>', unsafe_allow_html=True)
        st.caption("Market signals tailored for daily decision-makers and non-technical stakeholders.")

        # Top Executive Cards Row (4 Columns)
        c_sig, c_pnl, c_hold, c_risk = st.columns([1.3, 1.3, 1.4, 1.2])
        
        with c_sig:
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: {risk_summary['signal_color']};">
                <div style="color: #94a3b8; font-weight: 600; font-size: 0.85rem;">RECOMMENDED ACTION</div>
                <div style="color: {risk_summary['signal_color']}; font-size: 1.4rem; font-weight: 800; margin-top: 4px;">
                    {risk_summary['signal_badge']}
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px;">
                    4-Wk Target: <b>${risk_summary['target_4wk_price']:.2f}</b> ({risk_summary['delta_pct']:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_pnl:
            pnl_color = "#10b981" if risk_summary['pnl_100_barrels'] >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: {pnl_color};">
                <div style="color: #94a3b8; font-weight: 600; font-size: 0.85rem;">ESTIMATED PnL (100 BARRELS)</div>
                <div style="color: {pnl_color}; font-size: 1.6rem; font-weight: 800; margin-top: 4px;">
                    ${risk_summary['pnl_100_barrels']:+.2f}
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px;">
                    Per Barrel Change: <b>${risk_summary['delta_dollars']:+.2f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_hold:
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: #3b82f6;">
                <div style="color: #94a3b8; font-weight: 600; font-size: 0.85rem;">HOLDING & BREAKEVEN HORIZON</div>
                <div style="color: #60a5fa; font-size: 1.5rem; font-weight: 800; margin-top: 4px;">
                    Hold {risk_summary['holding_weeks']} Week(s)
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px;">
                    Est. Breakeven: <b>Week {risk_summary['holding_weeks']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_risk:
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: #f59e0b;">
                <div style="color: #94a3b8; font-weight: 600; font-size: 0.85rem;">MARKET RISK LEVEL</div>
                <div style="color: #f8fafc; font-size: 1.5rem; font-weight: 800; margin-top: 4px;">
                    {risk_summary['risk_level']}
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px;">
                    1-Wk Worst Dip: <b>-${risk_summary['var_95_dollars']:.2f}/bbl</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk & Directional Guardrail Warning Banner
        if risk_summary.get('has_asymmetric_risk') or risk_summary.get('has_low_dir_acc'):
            st.warning(f"🛡️ **Institutional Risk Filter**: {risk_summary['risk_guardrail_note']}")

        # Plain English Market Explanation
        st.info(f"💡 **Executive Overview**: {risk_summary['summary_text']}")

        # ---------------------------------------------------------------------
        # VERIFIABLE SOURCE CROSS-VERIFICATION & BENCHMARK ALIGNMENT
        # ---------------------------------------------------------------------
        with st.expander("🔗 **Verifiable Source Cross-Verification & Benchmark Alignment** (Click to inspect official external sources)", expanded=True):
            st.markdown(f"""
            <div style="background-color: #1e293b; border-left: 5px solid #10b981; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <span style="color: #10b981; font-weight: 700; font-size: 1.05rem;">{risk_summary['verification_status']}</span>
                <span style="color: #94a3b8; font-size: 0.9rem; margin-left: 12px;">
                    Model signal is verified against official macroeconomic forecasts and live financial exchange settlement curves.
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            v_col1, v_col2 = st.columns([2, 1])
            
            with v_col1:
                # Convert source metadata list into styled DataFrame
                sources_data = []
                for s in risk_summary['cross_verification_sources']:
                    sources_data.append({
                        "Verifiable Institution": s['Source'],
                        "Official Benchmark / Publication": s['Report'],
                        "Institutional Outlook": s['Benchmark Forecast'],
                        "Model Alignment": s['Alignment']
                    })
                sources_df = pd.DataFrame(sources_data)
                st.dataframe(sources_df, use_container_width=True, hide_index=True)

            with v_col2:
                st.markdown("""
                **Official Verifiable Data Sources**:
                - 🌐 [U.S. EIA STEO Official Report](https://www.eia.gov/outlooks/steo/)
                - 🌐 [ICE Brent Crude Futures Curve](https://www.theice.com/products/219/Brent-Crude-Futures)
                - 🌐 [OPEC Monthly Oil Market Report](https://www.opec.org/opec_web/en/publications/338.htm)
                - 🌐 [Yahoo Finance Brent (BZ=F)](https://finance.yahoo.com/quote/BZ%3DF/)
                
                *Cross-verification confidence is computed by measuring directional alignment between TSFP ML forecast outputs and U.S. EIA Short-Term Energy Outlook consensus projections.*
                """)
        st.markdown('<div class="section-header">🧮 4-Week Holding & Asset Value Simulator</div>', unsafe_allow_html=True)
        st.caption("Calculate expected asset value if you purchase Brent Crude today and hold over a 4-week horizon.")
        
        with st.container():
            col_calc_input, col_calc_res = st.columns([1, 2])
            
            with col_calc_input:
                calc_mode = st.radio("Simulation Input Type:", ["Number of Barrels", "Capital Invested ($)"], horizontal=True)
                
                if calc_mode == "Number of Barrels":
                    num_barrels = st.number_input("Enter Barrels to Buy:", min_value=1, max_value=10000, value=100, step=10)
                    initial_cost = num_barrels * risk_summary['latest_price']
                else:
                    capital_usd = st.number_input("Enter Capital to Invest ($):", min_value=100.0, max_value=1000000.0, value=5000.0, step=500.0)
                    num_barrels = capital_usd / risk_summary['latest_price']
                    initial_cost = capital_usd
                
                st.caption(f"**Current Investment Cost**: ${initial_cost:,.2f} ({num_barrels:.1f} barrels @ ${risk_summary['latest_price']:.2f}/bbl)")

            with col_calc_res:
                # Calculate weekly projected asset values
                w1_val = num_barrels * multi_horizon_preds[0]
                w2_val = num_barrels * multi_horizon_preds[1]
                w3_val = num_barrels * multi_horizon_preds[2]
                w4_val = num_barrels * multi_horizon_preds[3]
                
                w1_pnl = w1_val - initial_cost
                w2_pnl = w2_val - initial_cost
                w3_pnl = w3_val - initial_cost
                tot_pnl = w4_val - initial_cost
                roi_pct = (tot_pnl / initial_cost) * 100
                
                # Correct signed delta formatting for losses (negative values start with '-' to render in RED with down arrow ↓)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Week 1 Value", f"${w1_val:,.2f}", f"-${abs(w1_pnl):,.2f}" if w1_pnl < 0 else f"+${w1_pnl:,.2f}")
                mc2.metric("Week 2 Value", f"${w2_val:,.2f}", f"-${abs(w2_pnl):,.2f}" if w2_pnl < 0 else f"+${w2_pnl:,.2f}")
                mc3.metric("Week 3 Value", f"${w3_val:,.2f}", f"-${abs(w3_pnl):,.2f}" if w3_pnl < 0 else f"+${w3_pnl:,.2f}")
                
                w4_delta_str = f"-${abs(tot_pnl):,.2f} ({roi_pct:.1f}%)" if tot_pnl < 0 else f"+${tot_pnl:,.2f} ({roi_pct:+.1f}%)"
                mc4.metric("Week 4 Forecast", f"${w4_val:,.2f}", w4_delta_str)

        # ---------------------------------------------------------------------
        # HISTORICAL (PAST 4 WKS) + PROJECTED (NEXT 4 WKS) TRAJECTORY CHART
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-header">📈 Past 4-Week Actual & Projected 4-Week Price Trajectory</div>', unsafe_allow_html=True)
        st.caption("Seamless 8-week continuous timeline combining historical spot performance with champion model 4-week projected path.")
        
        # Historical Past 4 Weeks (5 points: t-4, t-3, t-2, t-1, t0)
        hist_5w = df_raw['Brent_Crude'].iloc[-5:]
        hist_dates = hist_5w.index
        hist_prices = hist_5w.values

        # Projected Next 4 Weeks (5 points: t0, t+1, t+2, t+3, t+4)
        last_date = df_raw.index[-1]
        horizon_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]
        chart_dates = [last_date] + horizon_dates
        chart_prices = [risk_summary['latest_price']] + multi_horizon_preds
        
        xgb_rmse = float(metrics_df[metrics_df['Model']=='XGBoost']['RMSE ($)'].iloc[0]) if not metrics_df[metrics_df['Model']=='XGBoost'].empty else 3.5
        upper_band = [p + (1.96 * xgb_rmse) for p in chart_prices]
        lower_band = [p - (1.96 * xgb_rmse) for p in chart_prices]

        fig_layman = go.Figure()
        
        # 1. Historical Actual Price Line (Past 4 Weeks)
        fig_layman.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            name='Historical Brent Price (Past 4 Wks)', mode='lines+markers',
            line=dict(color='#38bdf8', width=3.5),
            marker=dict(size=7, color='#7dd3fc'),
            hovertemplate='<b>Historical Date</b>: %{x|%Y-%m-%d}<br><b>Actual Price</b>: $%{y:.2f}<extra></extra>'
        ))

        # 2. Shaded Confidence Band (Next 4 Weeks)
        fig_layman.add_trace(go.Scatter(
            x=chart_dates, y=upper_band,
            name='Best Case (+1.96 Std)', line=dict(color='rgba(16, 185, 129, 0.2)', width=0), showlegend=False
        ))
        fig_layman.add_trace(go.Scatter(
            x=chart_dates, y=lower_band,
            name='Forecast Range (95% CI)', line=dict(color='rgba(16, 185, 129, 0.2)', width=0),
            fill='tonexty', fillcolor='rgba(16, 185, 129, 0.15)', showlegend=True
        ))
        
        # 3. Projected Forecast Path Line (Next 4 Weeks)
        fig_layman.add_trace(go.Scatter(
            x=chart_dates, y=chart_prices,
            name='Projected Price Path (Next 4 Wks)', mode='lines+markers',
            line=dict(color='#10b981', width=4, dash='solid'),
            marker=dict(size=8, color='#34d399'),
            hovertemplate='<b>Forecast Date</b>: %{x|%Y-%m-%d}<br><b>Projected Price</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add vertical line and annotation for Current Date Baseline
        fig_layman.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1, yref="paper",
            line=dict(color="#f59e0b", width=2, dash="dash")
        )
        fig_layman.add_annotation(
            x=last_date, y=1.02, yref="paper",
            text=f"📍 Spot Baseline (${risk_summary['latest_price']:.2f})",
            showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(color="#f59e0b", size=11, family="sans-serif")
        )

        fig_layman.update_xaxes(
            showgrid=True, gridcolor='#334155', gridwidth=1,
            showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='#94a3b8', spikethickness=1
        )
        fig_layman.update_yaxes(
            showgrid=True, gridcolor='#334155', gridwidth=1,
            showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='#94a3b8', spikethickness=1
        )
        fig_layman.update_layout(
            template='plotly_dark',
            height=440,
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#1e293b', font_size=13),
            xaxis_title="Continuous 8-Week Timeline (Past 4 Wks -> Current Spot -> Next 4 Wks)",
            yaxis_title="Price ($/Barrel)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_layman, use_container_width=True)

    # =========================================================================
    # VIEW MODE 2: QUANTITATIVE ANALYTICS (PROFESSIONAL AUDIENCE)
    # =========================================================================
    else:
        st.markdown('<div class="section-header">🔬 Quantitative Analytics & Multi-Model Evaluation</div>', unsafe_allow_html=True)
        
        # ---------------------------------------------------------------------
        # MODEL EXPLAINABILITY WATERFALL CHART ("Why this price?")
        # ---------------------------------------------------------------------
        st.markdown('### 🔍 Forecast Attribution Waterfall ("Why This Price?")')
        st.caption(
            "Deconstructs the exact dollar contribution of market drivers shifting the forecast relative to the baseline price.",
            help=(
                "ℹ️ **Market Driver & Price Lag Definitions**:\n\n"
                "• **Previous Week Close (Lag 1)**: Brent Crude price from 1 week ago (immediate price momentum).\n\n"
                "• **Secondary Price Lag (Lag 2)**: Brent Crude price from 2 weeks ago (short-term trend).\n\n"
                "• **Tertiary Price Lag (Lag 3)**: Brent Crude price from 3 weeks ago (medium-term autocorrelation).\n\n"
                "• **US Dollar Index (DXY)**: USD currency strength impact (oil is USD-denominated, creating an inverse relationship).\n\n"
                "• **VIX Volatility Index**: Global financial market risk and fear sentiment.\n\n"
                "• **OPEC+ Production Level**: Global crude oil supply quotas and output levels.\n\n"
            )
        )

        champ_model = model_results['models'].get(champion_col, model_results['models'].get('XGBoost', None))
        if champ_model is not None:
            last_row = df_engineered.iloc[-1]
            X_train = model_results['X_train']
            contrib_df, base_p, pred_p = calculate_feature_contributions(champ_model, last_row, X_train)
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Forecast Drivers",
                orientation="v",
                measure=["relative"] * len(contrib_df),
                x=contrib_df['Feature'],
                textposition="outside",
                text=[f"${v:+.2f}" for v in contrib_df['Contribution_Dollars']],
                y=contrib_df['Contribution_Dollars'],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}}
            ))
            
            fig_waterfall.update_layout(
                title=dict(text=f"Dollar Contributions to Forecast ({champion_col} Champion | Base Spot: ${base_p:.2f} -> Forecast: ${pred_p:.2f})", font=dict(size=14)),
                template='plotly_dark',
                height=380,
                yaxis_title="Impact ($/Barrel)"
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # MULTI-MODEL FORECAST & FAN CHART
        # ---------------------------------------------------------------------
        st.markdown('### 📈 Multi-Model Price Forecast Comparison')
        
        # Dynamic Year Filter Dropdown (filters available test set years dynamically)
        test_years = sorted(list(preds_df.index.year.unique()))
        year_options = ["All Test Period (Reset)"] + [str(y) for y in test_years]
        
        col_year, col_space = st.columns([1.5, 3])
        with col_year:
            selected_year = st.selectbox("📅 Filter Historical Test Chart by Year:", year_options, index=0)

        # Filter dataset based on selected year cleanly
        if selected_year != "All Test Period (Reset)":
            filtered_preds = preds_df[preds_df.index.year == int(selected_year)]
            if filtered_preds.empty:
                filtered_preds = preds_df
        else:
            filtered_preds = preds_df

        # Interactive Multi-Model Forecast Chart
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(
            x=filtered_preds.index, y=filtered_preds['Actual_Brent'],
            name='Actual Brent Price', line=dict(color='#f8fafc', width=3.5),
            hovertemplate='<b>Actual Brent</b>: $%{y:.2f}<extra></extra>'
        ))

        colors = {'SARIMAX': '#f59e0b', 'LASSO': '#3b82f6', 'Random Forest': '#a855f7', 'XGBoost': '#10b981', 'LightGBM': '#34d399', 'LSTM': '#ec4899'}
        
        model_cols = [c for c in filtered_preds.columns if c != 'Actual_Brent']
        for m in model_cols:
            fig_prof.add_trace(go.Scatter(
                x=filtered_preds.index, y=filtered_preds[m],
                name=m,
                line=dict(color=colors.get(m, '#94a3b8'), width=2, dash='dot' if m=='SARIMAX' else 'solid'),
                hovertemplate=f'<b>{m}</b>: $%{{y:.2f}}<extra></extra>'
            ))

        # Shaded 95% Confidence Bounds around Champion
        if champion_col in filtered_preds.columns:
            champ_rmse = float(metrics_df[metrics_df['Model']==champion_col]['RMSE ($)'].iloc[0]) if not metrics_df[metrics_df['Model']==champion_col].empty else 3.5
            upper_bound = filtered_preds[champion_col] + (1.96 * champ_rmse)
            lower_bound = filtered_preds[champion_col] - (1.96 * champ_rmse)
            
            fig_prof.add_trace(go.Scatter(
                x=filtered_preds.index, y=upper_bound,
                name=f'{champion_col} 95% Upper CI', line=dict(color='rgba(16, 185, 129, 0.2)', width=0), showlegend=False
            ))
            fig_prof.add_trace(go.Scatter(
                x=filtered_preds.index, y=lower_bound,
                name=f'{champion_col} 95% Lower CI', line=dict(color='rgba(16, 185, 129, 0.2)', width=0),
                fill='tonexty', fillcolor='rgba(16, 185, 129, 0.15)', showlegend=True
            ))

        fig_prof.update_xaxes(
            showgrid=True, gridcolor='#334155', gridwidth=1,
            showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='#94a3b8', spikethickness=1
        )
        fig_prof.update_yaxes(
            showgrid=True, gridcolor='#334155', gridwidth=1,
            showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='#94a3b8', spikethickness=1
        )
        fig_prof.update_layout(
            template='plotly_dark',
            height=480,
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#1e293b', font_size=13),
            title=dict(text=f"Multi-Model Out-of-Sample Evaluation ($/Barrel) - Filter: {selected_year}", font=dict(size=15)),
            xaxis=dict(title="Timeline"),
            yaxis=dict(title="Price ($/Barrel)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prof, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # PERFORMANCE SCOREBOARD & RISK METRICS
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-header">📊 Performance Scoreboard & Quantitative Risk Indicators</div>', unsafe_allow_html=True)
        
        # Display Champion Model Active Status Callout
        champ_score_val = metrics_df[metrics_df['Model']==champion_col]['Composite Score'].iloc[0] if (metrics_df is not None and not metrics_df[metrics_df['Model']==champion_col].empty) else 'N/A'
        champ_dir_val = metrics_df[metrics_df['Model']==champion_col]['Dir Accuracy (%)'].iloc[0] if (metrics_df is not None and not metrics_df[metrics_df['Model']==champion_col].empty) else 'N/A'
        st.info(f"🏆 **Active Champion Model**: **{champion_col}** | Composite Rank Score: **{champ_score_val}** | Directional Accuracy: **{champ_dir_val}%**")

        col_m_left, col_m_right = st.columns([2.2, 1])
        with col_m_left:
            st.markdown(
                "### 📋 Out-of-Sample Model Performance Matrix",
                help=(
                    "ℹ️ **Evaluation Metrics Definitions**:\n\n"
                    "• **Composite Score**: Normalized 0-100 score weighting Dir Accuracy (40%), RMSE (30%), and R² (30%).\n\n"
                    "• **MAE ($)**: Mean Absolute Error — average dollar forecast error per barrel.\n\n"
                    "• **RMSE ($)**: Root Mean Squared Error — square-root of average squared errors, penalizing larger deviations.\n\n"
                    "• **R² Score**: Coefficient of Determination — proportion of price variance explained by the model (1.0 = 100%).\n\n"
                    "• **MAPE (%)**: Mean Absolute Percentage Error — average percentage forecast error relative to actual price.\n\n"
                    "• **Dir Accuracy (%)**: Directional Accuracy — percentage of correct UP/DOWN price movement predictions.\n\n"
                    "• **Durbin-Watson**: Tests residual autocorrelation (2.0 = ideal zero autocorrelation, < 1.5 = positive autocorrelation).\n\n"
                    "• **Hyperparameters**: Algorithm tuning configuration."
                )
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
        with col_m_right:
            st.markdown(
                "### 🛡️ Value at Risk (VaR)",
                help=(
                    "ℹ️ **Value at Risk (VaR) Definition & Calculation**:\n\n"
                    "• **Definition**: Value at Risk measures the maximum expected loss per barrel over a holding period under normal market conditions at a given statistical confidence level.\n\n"
                    "• **Calculation Method**: Derived via historical simulation on weekly log price returns.\n\n"
                    "• **1-Week 95% VaR**: The 5th percentile worst weekly return loss (expected to occur 1 week out of 20).\n\n"
                    "• **1-Week 99% VaR**: The 1st percentile worst weekly return loss (expected to occur 1 week out of 100).\n\n"
                    "• **1-Month 95% VaR**: Scaled 4-week holding period risk derived via square-root of time rule (VaR_1wk × √4).\n\n"
                )
            )
            st.metric("1-Week 95% VaR", f"${risk_summary['var_95_dollars']:.2f} / bbl")
            st.metric("1-Week 99% VaR", f"${risk_summary['var_99_dollars']:.2f} / bbl")
            st.metric("1-Month 95% VaR", f"${risk_summary['var_1mo_95_dollars']:.2f} / bbl")

        # ---------------------------------------------------------------------
        # FEATURE IMPORTANCE & RESIDUAL DIAGNOSTICS
        # ---------------------------------------------------------------------
        col_fi, col_res = st.columns(2)
        with col_fi:
            st.markdown("### 🔍 Tree Model Feature Importance Breakdown")
            
            # Map raw feature variable strings to clean, human-readable domain labels
            feature_map = {
                'Price_Lag_1': 'Previous Week Close (Lag 1)',
                'Price_Lag_2': 'Secondary Price Lag (Lag 2)',
                'Price_Lag_3': 'Tertiary Price Lag (Lag 3)',
                'USD_Z': 'US Dollar Index (DXY)',
                'VIX_Z': 'VIX Volatility Index',
                'OPEC_Z': 'OPEC+ Production Level'
            }
            
            fi_df = model_results['feature_importances'].copy()
            fi_df['Feature'] = fi_df['Feature'].map(feature_map).fillna(fi_df['Feature'])
            
            fig_fi = px.bar(
                fi_df, x='Feature', y=['XGBoost', 'Random Forest'],
                barmode='group', title="Feature Importance (XGBoost vs Random Forest)",
                template='plotly_dark'
            )
            fig_fi.update_layout(height=350)
            st.plotly_chart(fig_fi, use_container_width=True)
            
        with col_res:
            st.markdown(
                "### 📊 Forecast Error Residual Distribution",
                help=(
                    "ℹ️ **Forecast Error Residual Diagnostics**:\n\n"
                    "• **Definition**: Residual error is the difference between actual price and model predicted price ($e_t = y_t - \\hat{y}_t$).\n\n"
                    "• **Interpretation**: A symmetrical bell-curve distribution centered near $0.00 with low standard deviation indicates an unbiased forecasting model without systematic over- or under-forecasting bias.\n\n"
                )
            )
            residuals = preds_df['Actual_Brent'] - preds_df['XGBoost']
            res_mean = residuals.mean()
            res_std = residuals.std()
            
            fig_res = px.histogram(
                residuals, nbins=25, title=f"Residual Errors (Mean: ${res_mean:.2f}, Std: ${res_std:.2f})",
                labels={'value': 'Error ($)'}, template='plotly_dark', color_discrete_sequence=['#38bdf8']
            )
            fig_res.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)

        # ---------------------------------------------------------------------
        # ADVANCED ANALYSIS EXPANDERS (PROFESSIONAL MODE EXCLUSIVE)
        # ---------------------------------------------------------------------
        st.markdown("---")
        # --- Upload Custom Data Section (Commented Out) ---
        # with st.expander("📁 Upload Custom Data & Run Predictions"):
        #     uploaded_file = st.file_uploader("Upload CSV or Excel File:", type=['csv', 'xlsx'])
        #     if uploaded_file is not None:
        #         user_df, err = process_uploaded_file(uploaded_file)
        #         if err:
        #             st.error(err)
        #         else:
        #             st.success(f"Successfully loaded `{uploaded_file.name}` ({len(user_df)} rows).")
        #             st.dataframe(user_df.head(10), use_container_width=True)

        with st.expander("🎛️ Macroeconomic What-If Stress Tester"):
            st.markdown(
                "Simulate how shocks in US Dollar Strength, Volatility, and OPEC Production impact Brent Crude prices.",
                help=(
                    "ℹ️ **How the Stress Tester Works & Scenario Guide**:\n\n"
                    "• **Weekly Friday Forecast Prices**: All baseline and scenario prices shown are **forecasted weekly Friday closing prices** (Weeks 1, 2, 3, and 4 ahead) generated by the AI Champion Model starting from the latest weekly Friday market price.\n\n"
                    "• **Why Sliders Move in Both Directions (Positive & Negative %)**:\n\n"
                    "  Macroeconomic indicators can move up or down relative to today's market level:\n\n"
                    "  - **💵 US Dollar Index (DXY)**:\n"
                    "    - *Rate Hikes / Dollar Surge*: **+2% to +10%** (e.g., DXY rises from 100.80 to 105.84).\n"
                    "    - *Rate Cuts / Dollar Weakening*: **-2% to -10%** (e.g., DXY drops from 100.80 to 95.76).\n\n"
                    "  - **🛢️ OPEC+ Production**:\n"
                    "    - *Supply Cuts*: **-3% to -10%** (e.g., OPEC cuts output from 27.5 mb/d to 25.0 mb/d).\n"
                    "    - *Production Increases*: **+3% to +10%** (e.g., OPEC expands output to 29.0 mb/d).\n\n"
                    "  - **📈 VIX Volatility Index**:\n"
                    "    - *Market Panic / Geopolitical Conflict*: **+20% to +50%** (VIX spikes).\n"
                    "    - *Calm Market / Economic Rally*: **-10% to -30%** (VIX cools down).\n\n"
                    "  *(Sliders span both positive and negative percentages so you can simulate both supply cuts and production hikes, or dollar rallies and devaluations).* \n\n"
                    "• **Recommended Realistic Scenario Presets**:\n\n"
                    "  - **Scenario A (Hawkish Fed / Dollar Surge)**: DXY +3.0%, VIX +15.0%, OPEC 0.0%\n"
                    "  - **Scenario B (Geopolitical Crisis / Supply Shock)**: DXY -2.0%, VIX +30.0%, OPEC -4.0%\n"
                    "  - **Scenario C (Global Expansion / Commodity Rally)**: DXY -3.0%, VIX -10.0%, OPEC +2.0%\n"
                )
            )
            last_row = df_engineered.iloc[-1]
            
            col_s1, col_s2, col_s3 = st.columns(3)
            
            curr_dxy = float(last_row['US_Dollar_Index']) if 'US_Dollar_Index' in last_row else 100.80
            curr_vix = float(last_row['VIX_Index']) if 'VIX_Index' in last_row else 20.66
            curr_opec = float(last_row['OPEC_Plus_Production']) if 'OPEC_Plus_Production' in last_row else 27.52
            
            with col_s1:
                usd_s = st.slider("💵 US Dollar Index (DXY) Shock (%):", -20.0, 20.0, 0.0, 1.0)
                sim_dxy = curr_dxy * (1.0 + usd_s / 100.0)
                st.caption(f"DXY Value: `{curr_dxy:.2f}` → **`{sim_dxy:.2f}`** ({usd_s:+.1f}%)")
                
            with col_s2:
                vix_s = st.slider("📈 VIX Volatility Shock (%):", -50.0, 50.0, 0.0, 5.0)
                sim_vix = curr_vix * (1.0 + vix_s / 100.0)
                st.caption(f"VIX Value: `{curr_vix:.2f}` → **`{sim_vix:.2f}`** ({vix_s:+.1f}%)")
                
            with col_s3:
                opec_s = st.slider("🛢️ OPEC Production Shock (%):", -15.0, 15.0, 0.0, 1.0)
                sim_opec = curr_opec * (1.0 + opec_s / 100.0)
                st.caption(f"OPEC Output: `{curr_opec:.2f} mb/d` → **`{sim_opec:.2f} mb/d`** ({opec_s:+.1f}%)")
            
            champion_col = model_results.get('champion_col', 'XGBoost')
            active_model = model_results['models'].get(champion_col, model_results['models'].get('XGBoost'))
            if active_model:
                sim_out = simulate_scenario(active_model, last_row, usd_s, vix_s, opec_s, df_engineered=df_engineered)
                scenario_4w = sim_out['multi_horizon_scenario']
                base_price = sim_out['base_price']
                
                st.markdown("#### 📅 4-Week Horizon Scenario Forecast")
                st.caption(f"Simulated price trajectory using the champion model (**{champion_col}**) under your macro shock scenario:")
                
                w_cols = st.columns(4)
                for idx, w_price in enumerate(scenario_4w):
                    d_dollars = w_price - base_price
                    d_pct = (d_dollars / base_price) * 100
                    d_str = f"-${abs(d_dollars):.2f} ({d_pct:.2f}%)" if d_dollars < 0 else f"+${d_dollars:.2f} ({d_pct:.2f}%)"
                    w_cols[idx].metric(
                        f"Week {idx + 1} Scenario",
                        f"${w_price:.2f}",
                        d_str
                    )
                
                # 4-Week Comparison Chart (Baseline vs Scenario)
                weeks_labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
                fig_scenario = go.Figure()
                
                # Baseline 4-Week Projection
                if multi_horizon_preds and len(multi_horizon_preds) == 4:
                    fig_scenario.add_trace(go.Scatter(
                        x=weeks_labels,
                        y=multi_horizon_preds,
                        mode='lines+markers',
                        name='Baseline 4-Wk Forecast',
                        line=dict(color='#94a3b8', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                
                # What-If Scenario Projection
                fig_scenario.add_trace(go.Scatter(
                    x=weeks_labels,
                    y=scenario_4w,
                    mode='lines+markers',
                    name='What-If Scenario Forecast',
                    line=dict(color='#38bdf8', width=3),
                    marker=dict(size=10, symbol='diamond')
                ))
                
                fig_scenario.update_layout(
                    title=f"4-Week Price Trajectory Comparison ({champion_col})",
                    xaxis_title="Forecast Horizon",
                    yaxis_title="Brent Price ($)",
                    template="plotly_dark",
                    height=330,
                    margin=dict(l=40, r=40, t=50, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_scenario, use_container_width=True)

# Run app
if __name__ == '__main__':
    main()

import os
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_processed')

def pull_market_data(start_date="2016-01-01", end_date=None):
    """Fetch market data from Yahoo Finance and resample to weekly Friday close."""
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    tickers = ["BZ=F", "CL=F", "DX-Y.NYB", "^VIX"]
    raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    raw_data.columns = ["Brent_Crude", "WTI_Crude", "US_Dollar_Index", "VIX_Index"]
    weekly_data = raw_data.resample('W-FRI').last()
    cleaned_data = weekly_data.ffill().bfill()
    return cleaned_data

def pull_opec_data():
    """Fetch OPEC production data from EIA API or fallback if offline."""
    eia_api_key = "1aVwQiijaPZ9PJde4Pfuf1AU5UMH9ARVbRlIbmgD"
    series_id = 'COPR_OPECPLUS'
    url = f"https://api.eia.gov/v2/steo/data/?api_key={eia_api_key}&frequency=monthly&data[0]=value&facets[seriesId][]={series_id}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            opec_df = pd.DataFrame(data['response']['data'])
            opec_df = opec_df[['period', 'value']].copy()
            opec_df.rename(columns={'value': 'OPEC_Plus_Production'}, inplace=True)
            opec_df['OPEC_Plus_Production'] = opec_df['OPEC_Plus_Production'].astype(float)
            opec_df['period'] = pd.to_datetime(opec_df['period'] + '-01')
            opec_df.set_index('period', inplace=True)
            opec_df.sort_index(inplace=True)
            return opec_df
    except Exception:
        pass
    return None

def build_master_dataset(start_date="2016-01-01", end_date=None, force_refresh=False):
    """Build master dataset combining financial indicators and OPEC production."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    cache_name = "master_dataset_cache.csv" if start_date == "2016-01-01" and end_date is None else f"master_dataset_{start_date[:4]}_{'now' if end_date is None else end_date[:4]}.csv"
    cache_path = os.path.join(DATA_PROCESSED_DIR, cache_name)
    
    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception:
            pass

    financial_df = pull_market_data(start_date=start_date, end_date=end_date)
    opec_df = pull_opec_data()
    
    if opec_df is not None and not opec_df.empty:
        master_dataset = pd.merge(financial_df, opec_df, left_index=True, right_index=True, how='outer')
        master_dataset['OPEC_Plus_Production'] = master_dataset['OPEC_Plus_Production'].ffill().bfill()
        master_dataset = master_dataset.loc[financial_df.index.min() : financial_df.index.max()]
    else:
        master_dataset = financial_df.copy()
        master_dataset['OPEC_Plus_Production'] = 41.5

    master_dataset = master_dataset.dropna(subset=['Brent_Crude']).copy()
    master_dataset.to_csv(cache_path)
    return master_dataset

def engineer_features(df, target_col='Brent_Crude'):
    """Engineer log features, Z-score normalization, moving averages, and lags for Brent or WTI."""
    data = df.copy()
    
    # Target log transformation
    target_log_name = 'Brent_Log' if target_col == 'Brent_Crude' else 'Target_Log'
    data[target_log_name] = np.log1p(data[target_col])
    data['Brent_Log'] = data[target_log_name]
    
    # Moving Averages
    data['SMA_4'] = data[target_col].rolling(window=4).mean()
    data['EMA_4'] = data[target_col].ewm(span=4, adjust=False).mean()
    data['Median_4'] = data[target_col].rolling(window=4).median()
    
    # Z-score normalization
    scaler = StandardScaler()
    other_oil = 'WTI_Crude' if target_col == 'Brent_Crude' else 'Brent_Crude'
    z_cols = [other_oil, 'US_Dollar_Index', 'VIX_Index', 'OPEC_Plus_Production']
    z_new_names = ['WTI_Z', 'USD_Z', 'VIX_Z', 'OPEC_Z']
    data[z_new_names] = scaler.fit_transform(data[z_cols])
    
    # Autoregressive Lags
    data['Price_Lag_1'] = data[target_log_name].shift(1)
    data['Price_Lag_2'] = data[target_log_name].shift(2)
    data['Price_Lag_3'] = data[target_log_name].shift(3)
    
    data = data.dropna().copy()
    return data

def process_uploaded_file(uploaded_file):
    """Parse custom CSV or Excel file uploaded by user into DataFrame."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel (.xlsx)."

        # Try setting date index if available
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
            df.sort_index(inplace=True)
            
        return df, None
    except Exception as e:
        return None, f"Failed to parse file: {str(e)}"

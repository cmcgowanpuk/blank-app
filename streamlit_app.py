import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit page configuration
st.set_page_config(
    page_title="UK Electricity Generation & Interconnector Flows",
    page_icon="⚡",
    layout="wide"
)

# Dictionary for generation type display names
GENERATION_DISPLAY_NAMES = {
    'BIOMASS': 'Biomass',
    'COAL': 'Coal',
    'CCGT': 'CCGT',
    'NPSHYD': 'Non-PS Hydro',
    'NUCLEAR': 'Nuclear',
    'OCGT': 'OCGT',
    'OIL': 'Oil',
    'OTHER': 'Other',
    'PS': 'Pumped Storage',
    'WIND': 'Wind',
    'SOLAR': 'Solar'
}

# Cache the data fetching functions to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fuelinst_data(start_time, end_time):
    """
    Fetch FUELINST data from BMRS API for generation types
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary"
    
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "format": "json"
    }
    
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_interconnector_data(settlement_date_from, settlement_date_to):
    """
    Fetch interconnector flow data from BMRS API
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/interconnectors"
    
    params = {
        "settlementDateFrom": settlement_date_from,
        "settlementDateTo": settlement_date_to,
        "format": "json"
    }
    
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        return None

def process_generation_data(raw_data):
    """
    Process the raw FUELINST data into a DataFrame for generation types only
    """
    all_records = []
    
    interconnectors = ['INTELEC', 'INTEW', 'INTFR', 'INTGRNL', 'INTIFA2', 
                      'INTIRL', 'INTNED', 'INTNEM', 'INTNSL', 'INTVKL']
    
    for period in raw_data:
        start_time = period['startTime']
        settlement_period = period['settlementPeriod']
        
        for fuel_data in period['data']:
            if fuel_data['fuelType'] not in interconnectors:
                # Map to display name
                fuel_type = fuel_data['fuelType']
                display_name = GENERATION_DISPLAY_NAMES.get(fuel_type, fuel_type)
                
                record = {
                    'startTime': start_time,
                    'settlementPeriod': settlement_period,
                    'fuelType': display_name,
                    'generation': fuel_data['generation']
                }
                all_records.append(record)
    
    df = pd.DataFrame(all_records)
    if not df.empty:
        df['startTime'] = pd.to_datetime(df['startTime'])
    
    return df

def process_interconnector_data(raw_data):
    """
    Process interconnector data with proper import/export signs
    """
    all_records = []
    
    # Mapping from API names to display names with countries
    interconnector_mapping = {
        'Eleclink (INTELEC)': 'Eleclink (France)',
        'Ireland(East-West)': 'East-West (Ireland)',
        'France(IFA)': 'IFA (France)',
        'IFA2 (INTIFA2)': 'IFA2 (France)',
        'Northern Ireland(Moyle)': 'Moyle (Northern Ireland)',
        'Netherlands(BritNed)': 'BritNed (Netherlands)',
        'Belgium (Nemolink)': 'Nemolink (Belgium)',
        'North Sea Link (INTNSL)': 'North Sea Link (Norway)',
        'Viking Link (INTVKL)': 'Viking Link (Denmark)',
        'Greenlink (INTGRNL)': 'Greenlink (Ireland)'
    }
    
    if raw_data and 'data' in raw_data:
        for period in raw_data['data']:
            start_time = period.get('startTime')
            settlement_date = period.get('settlementDate')
            settlement_period = period.get('settlementPeriod')
            interconnector_name = period.get('interconnectorName')
            generation = period.get('generation', 0)
            
            # Map to display name with country
            fuel_type = interconnector_mapping.get(interconnector_name, interconnector_name)
            
            record = {
                'startTime': start_time,
                'settlementDate': settlement_date,
                'settlementPeriod': settlement_period,
                'fuelType': fuel_type,
                'generation': generation
            }
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    if not df.empty:
        df['startTime'] = pd.to_datetime(df['startTime'])
    
    return df

def calculate_peak_offpeak_weekly(df):
    """
    Calculate Peak and Off-Peak MWh per fuel type per week
    """
    if df.empty:
        return pd.DataFrame()
    
    df['hour'] = df['startTime'].dt.hour
    df['day_of_week'] = df['startTime'].dt.dayofweek
    df['year'] = df['startTime'].dt.year
    df['week_number'] = df['startTime'].dt.isocalendar().week
    
    df['week_start'] = df['startTime'] - pd.to_timedelta(df['startTime'].dt.dayofweek, unit='d')
    df['week_start'] = df['week_start'].dt.floor('D')
    
    df['efa_week'] = 'Week ' + df['week_number'].astype(str).str.zfill(2) + '-' + df['year'].astype(str).str[-2:]
    
    df['is_peak'] = (
        (df['day_of_week'] < 5) &
        (df['hour'] >= 7) & 
        (df['hour'] < 19)
    )
    
    df['mwh'] = df['generation'] * 0.5
    df['period_type'] = df['is_peak'].apply(lambda x: 'Peak' if x else 'Off-Peak')
    
    weekly_mwh = df.groupby(['fuelType', 'week_start', 'efa_week', 'period_type'])['mwh'].sum().reset_index()
    
    return weekly_mwh

def calculate_total_interconnector_flow(int_df):
    """
    Calculate total interconnector flow per time period
    """
    if int_df.empty:
        return pd.DataFrame()
    
    total_flow = int_df.groupby(['startTime', 'settlementPeriod']).agg({
        'generation': 'sum'
    }).reset_index()
    
    total_flow['fuelType'] = 'TOTAL_INTERCONNECTOR'
    
    return total_flow

def create_plotly_chart(weekly_data, chart_type='generation'):
    """
    Create Plotly charts for either generation or interconnectors
    """
    if weekly_data.empty:
        return None
    
    # Convert MWh to TWh for display (divide by 1,000,000)
    weekly_data_twh = weekly_data.copy()
    weekly_data_twh['twh'] = weekly_data_twh['mwh'] / 1_000_000
    
    # Dark mode colors - Claude's orange and white
    peak_color = '#FF8C42'  # Claude's orange
    off_peak_color = '#E8E8E8'  # Light gray/white for off-peak
    
    if chart_type == 'generation':
        # Define the order for generation types
        generation_order = ['Biomass', 'Coal', 'CCGT', 'Non-PS Hydro', 'Nuclear', 
                          'OCGT', 'Oil', 'Other', 'Pumped Storage', 'Solar', 'Wind']
        fuel_types = [ft for ft in generation_order if ft in weekly_data_twh['fuelType'].unique()]
        # Add any missing types at the end
        for ft in sorted(weekly_data_twh['fuelType'].unique()):
            if ft not in fuel_types:
                fuel_types.append(ft)
        title = "Generation Types - Weekly Energy Production (TWh)"
        y_label = "TWh"
        y_range = [0, 1.5]  # Fixed range for generation
    else:
        # Define the order for interconnectors
        interconnector_order = ['Nemolink (Belgium)', 'Viking Link (Denmark)', 
                               'Eleclink (France)', 'IFA (France)', 'IFA2 (France)',
                               'East-West (Ireland)', 'Greenlink (Ireland)',
                               'BritNed (Netherlands)', 'North Sea Link (Norway)',
                               'Moyle (Northern Ireland)']
        individual_interconnectors = [ft for ft in interconnector_order 
                                     if ft in weekly_data_twh['fuelType'].unique() and ft != 'TOTAL_INTERCONNECTOR']
        fuel_types = individual_interconnectors
        if 'TOTAL_INTERCONNECTOR' in weekly_data_twh['fuelType'].unique():
            fuel_types.append('TOTAL_INTERCONNECTOR')
        title = "Interconnectors - Weekly Energy Flow (TWh)"
        y_label = "TWh (Import+/Export-)"
        y_range = [-0.1, 1.5]  # Fixed range for interconnectors
    
    rows = (len(fuel_types) + 1) // 2
    
    subplot_titles = []
    for ft in fuel_types:
        if ft == 'TOTAL_INTERCONNECTOR':
            subplot_titles.append('TOTAL NET FLOW')
        else:
            subplot_titles.append(ft)
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    for idx, fuel_type in enumerate(fuel_types):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        fuel_data = weekly_data_twh[weekly_data_twh['fuelType'] == fuel_type]
        
        pivot_data = fuel_data.pivot_table(
            index=['week_start', 'efa_week'],
            columns='period_type',
            values='twh',
            fill_value=0
        ).reset_index()
        
        if 'Off-Peak' in pivot_data.columns:
            fig.add_trace(
                go.Bar(
                    x=pivot_data['week_start'],
                    y=pivot_data['Off-Peak'],
                    name='Off-Peak' if idx == 0 else None,
                    marker_color=off_peak_color,
                    showlegend=True if idx == 0 else False,
                    customdata=pivot_data['efa_week'],
                    hovertemplate=f'{fuel_type} Off-Peak<br>%{{customdata}}<br>TWh: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        if 'Peak' in pivot_data.columns:
            fig.add_trace(
                go.Bar(
                    x=pivot_data['week_start'],
                    y=pivot_data['Peak'],
                    name='Peak' if idx == 0 else None,
                    marker_color=peak_color,
                    showlegend=True if idx == 0 else False,
                    customdata=pivot_data['efa_week'],
                    hovertemplate=f'{fuel_type} Peak<br>%{{customdata}}<br>TWh: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Set y-axis range for each subplot
        if chart_type == 'interconnector':
            fig.update_yaxes(
                title_text=y_label,
                range=y_range,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                row=row, col=col
            )
        else:
            fig.update_yaxes(
                title_text=y_label,
                range=y_range,
                row=row, col=col
            )
    
    fig.update_layout(
        barmode='stack',
        title=title,
        height=300 * rows,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#0E1117',
        font=dict(color='white')
    )
    
    return fig

def fetch_data(start_date, end_date, progress_bar):
    """
    Fetch data with progress updates
    """
    all_gen_data = []
    all_int_df = []
    current_date = start_date
    
    total_days = (end_date - start_date).days
    processed_days = 0
    
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=30), end_date)
        
        chunk_start_str = current_date.strftime("%Y-%m-%dT00:00:00Z")
        chunk_end_str = chunk_end.strftime("%Y-%m-%dT23:59:59Z")
        
        gen_chunk = get_fuelinst_data(chunk_start_str, chunk_end_str)
        if gen_chunk:
            all_gen_data.extend(gen_chunk)
        
        int_current = current_date
        while int_current <= chunk_end:
            int_end = min(int_current + timedelta(days=6), chunk_end)
            
            int_start_str = int_current.strftime("%Y-%m-%d")
            int_end_str = int_end.strftime("%Y-%m-%d")
            
            int_chunk = get_interconnector_data(int_start_str, int_end_str)
            if int_chunk:
                int_df = process_interconnector_data(int_chunk)
                if not int_df.empty:
                    all_int_df.append(int_df)
            
            int_current = int_end + timedelta(days=1)
        
        processed_days = min((chunk_end - start_date).days, total_days)
        progress_bar.progress(processed_days / total_days)
        
        current_date = chunk_end + timedelta(days=1)
    
    return all_gen_data, all_int_df

# Main Streamlit App
def main():
    st.title("UK Electricity Generation & Interconnector Flows")
    st.markdown("Analysis of UK electricity generation by fuel type and interconnector flows (imports/exports)")
    
    # Sidebar for date selection
    st.sidebar.header("Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1),
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now()
        )
    
    if st.sidebar.button("Fetch Data", type="primary"):
        with st.spinner("Fetching data from BMRS API..."):
            progress_bar = st.progress(0)
            
            # Fetch data
            all_gen_data, all_int_df = fetch_data(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time()),
                progress_bar
            )
            
            progress_bar.empty()
            
            # Process data
            gen_df = process_generation_data(all_gen_data)
            int_df = pd.concat(all_int_df, ignore_index=True) if all_int_df else pd.DataFrame()
            
            if not int_df.empty:
                total_flow_df = calculate_total_interconnector_flow(int_df)
                int_df_with_total = pd.concat([int_df, total_flow_df], ignore_index=True)
            else:
                int_df_with_total = int_df
            
            # Calculate weekly data
            gen_weekly = calculate_peak_offpeak_weekly(gen_df)
            int_weekly = calculate_peak_offpeak_weekly(int_df_with_total)
            
            # Store in session state
            st.session_state['gen_weekly'] = gen_weekly
            st.session_state['int_weekly'] = int_weekly
            st.session_state['data_loaded'] = True
    
    # Display charts if data is loaded
    if st.session_state.get('data_loaded', False):
        tab1, tab2 = st.tabs(["Generation", "Interconnectors"])
        
        with tab1:
            st.header("Generation by Fuel Type")
            if not st.session_state['gen_weekly'].empty:
                fig_gen = create_plotly_chart(st.session_state['gen_weekly'], 'generation')
                st.plotly_chart(fig_gen, use_container_width=True)
            else:
                st.warning("No generation data available")
        
        with tab2:
            st.header("Interconnector Flows")
            st.info("**Positive values** = Imports to UK | **Negative values** = Exports from UK")
            if not st.session_state['int_weekly'].empty:
                fig_int = create_plotly_chart(st.session_state['int_weekly'], 'interconnector')
                st.plotly_chart(fig_int, use_container_width=True)
            else:
                st.warning("No interconnector data available")
            
            # Download option
            if not st.session_state['gen_weekly'].empty or not st.session_state['int_weekly'].empty:
                all_weekly = pd.concat([st.session_state['gen_weekly'], st.session_state['int_weekly']], ignore_index=True)
                csv = all_weekly.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"uk_electricity_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("Select date range and click 'Fetch Data' to begin")

if __name__ == "__main__":
    main()

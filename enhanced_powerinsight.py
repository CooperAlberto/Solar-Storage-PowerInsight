import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="⚡ PowerInsight - Energy Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #d1edff;
        border-left: 4px solid #0084ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .battery-box {
        background: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def convert_24_to_12_hour(hour_24):
    """Convert 24-hour format to 12-hour format for display"""
    if hour_24 == 0:
        return "12:00 AM"
    elif hour_24 < 12:
        return f"{hour_24}:00 AM"
    elif hour_24 == 12:
        return "12:00 PM"
    else:
        return f"{hour_24 - 12}:00 PM"

def convert_12_to_24_hour(hour_12, period):
    """Convert 12-hour format back to 24-hour format"""
    if period == "AM":
        if hour_12 == 12:
            return 0
        else:
            return hour_12
    else:  # PM
        if hour_12 == 12:
            return 12
        else:
            return hour_12 + 12

# ============= USAGE ANALYSIS FUNCTIONS =============

def validate_usage_data(df):
    """Validate the electrical usage data"""
    errors = []
    warnings = []
    
    required_columns = ['Start Datetime', 'Net Usage']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
    
    if not errors:
        if df['Start Datetime'].isnull().any():
            warnings.append("Some datetime values are missing")
        if df['Net Usage'].isnull().any():
            warnings.append("Some usage values are missing")
        if (df['Net Usage'] < 0).any():
            warnings.append("Negative usage values detected")
        if (df['Net Usage'] > 50).any():
            warnings.append("Unusually high usage values detected (>50 kWh)")
    
    return errors, warnings

def clean_usage_data(df):
    """Clean and prepare the usage data"""
    df.columns = df.columns.str.strip()
    df['Start Datetime'] = pd.to_datetime(df['Start Datetime'], errors='coerce')
    df = df.dropna(subset=['Start Datetime', 'Net Usage'])
    df['Net Usage'] = pd.to_numeric(df['Net Usage'], errors='coerce')
    df = df.dropna(subset=['Net Usage'])
    
    # Extract time components
    df['Month'] = df['Start Datetime'].dt.month
    df['Day'] = df['Start Datetime'].dt.day
    df['Hour'] = df['Start Datetime'].dt.hour
    df['DayOfWeek'] = df['Start Datetime'].dt.day_name()
    df['Date'] = df['Start Datetime'].dt.date
    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Fall', 10: 'Fall', 11: 'Fall'})
    
    return df.sort_values('Start Datetime')

def generate_usage_insights(df, monthly_analysis):
    """Generate intelligent insights from the usage data"""
    insights = []
    
    # Peak usage month
    peak_month = max(monthly_analysis, key=lambda x: x['Total Usage'])
    low_month = min(monthly_analysis, key=lambda x: x['Total Usage'])
    
    insights.append(f"🔥 **Peak Usage**: {peak_month['Month']} had the highest consumption at {peak_month['Total Usage']:.1f} kWh")
    insights.append(f"🌟 **Lowest Usage**: {low_month['Month']} had the lowest consumption at {low_month['Total Usage']:.1f} kWh")
    
    # Day vs Night comparison
    total_day = sum([x['Average Daytime Usage'] for x in monthly_analysis])
    total_night = sum([x['Average Nighttime Usage'] for x in monthly_analysis])
    
    if total_day > total_night:
        insights.append(f"☀️ **Usage Pattern**: You use {((total_day/total_night - 1) * 100):.1f}% more electricity during the day")
    else:
        insights.append(f"🌙 **Usage Pattern**: You use {((total_night/total_day - 1) * 100):.1f}% more electricity at night")
    
    # Seasonal analysis
    seasonal_usage = df.groupby('Season')['Net Usage'].sum().to_dict()
    peak_season = max(seasonal_usage, key=seasonal_usage.get)
    insights.append(f"❄️🌞 **Seasonal Peak**: {peak_season} shows the highest energy consumption")
    
    # Weekly pattern
    weekend_usage = df[df['DayOfWeek'].isin(['Saturday', 'Sunday'])]['Net Usage'].mean()
    weekday_usage = df[~df['DayOfWeek'].isin(['Saturday', 'Sunday'])]['Net Usage'].mean()
    
    if weekend_usage > weekday_usage:
        insights.append(f"🏠 **Weekend Effect**: Usage is {((weekend_usage/weekday_usage - 1) * 100):.1f}% higher on weekends")
    else:
        insights.append(f"🏢 **Weekday Pattern**: Usage is {((weekday_usage/weekend_usage - 1) * 100):.1f}% higher on weekdays")
    
    return insights

def create_seasonal_chart(df):
    """Create seasonal usage comparison"""
    seasonal_data = df.groupby(['Season', 'Hour'])['Net Usage'].mean().reset_index()
    
    fig = px.line(
        seasonal_data, 
        x='Hour', 
        y='Net Usage', 
        color='Season',
        title="🌿 Seasonal Usage Patterns Throughout the Day",
        labels={'Net Usage': 'Average Usage (kWh)', 'Hour': 'Hour of Day'}
    )
    
    fig.update_layout(height=400)
    return fig

# ============= SOLAR+STORAGE ANALYSIS FUNCTIONS =============

def validate_solar_data(df):
    """Validate the solar production data with improved column detection"""
    errors = []
    warnings = []
    
    # Check for separated date/time columns (Month, Day, Hour format)
    has_month = any('month' in str(col).lower() for col in df.columns)
    has_day = any('day' in str(col).lower() for col in df.columns)
    has_hour = any('hour' in str(col).lower() for col in df.columns)
    
    if has_month and has_day and has_hour:
        # Find the actual column names
        month_col = next((col for col in df.columns if 'month' in str(col).lower()), None)
        day_col = next((col for col in df.columns if 'day' in str(col).lower()), None)
        hour_col = next((col for col in df.columns if 'hour' in str(col).lower()), None)
        
        # Find production column
        production_keywords = ['production', 'generation', 'solar', 'kwh', 'energy', 'power', 'output', 'yield']
        production_col = None
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(keyword in col_lower for keyword in production_keywords):
                production_col = col
                break
        
        if not production_col:
            # Look for numeric columns (excluding date/time columns)
            exclude_cols = [month_col, day_col, hour_col]
            for col in df.columns:
                if col not in exclude_cols:
                    try:
                        pd.to_numeric(df[col].dropna().head(10))
                        production_col = col
                        break
                    except:
                        continue
        
        if not production_col:
            errors.append(f"Solar file missing production column. Available columns: {list(df.columns)}")
        
        return errors, warnings, (month_col, day_col, hour_col), production_col
    
    else:
        # Original logic for datetime columns
        datetime_keywords = ['datetime', 'time', 'date', 'timestamp', 'ts', 'dt']
        datetime_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(keyword in col_lower for keyword in datetime_keywords):
                datetime_cols.append(col)
        
        if not datetime_cols:
            for col in df.columns:
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    datetime_like = True
                    for val in sample_values:
                        try:
                            pd.to_datetime(str(val))
                        except:
                            datetime_like = False
                            break
                    if datetime_like:
                        datetime_cols.append(col)
                        break
        
        # Find production column
        production_keywords = ['production', 'generation', 'solar', 'kwh', 'energy', 'power', 'output', 'yield', 'ac', 'dc']
        production_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(keyword in col_lower for keyword in production_keywords):
                production_cols.append(col)
        
        if not production_cols:
            for col in df.columns:
                if col not in datetime_cols:
                    try:
                        pd.to_numeric(df[col].dropna().head(10))
                        production_cols.append(col)
                        break
                    except:
                        continue
        
        # Error checking
        if not datetime_cols:
            errors.append(f"Solar file missing datetime column. Available columns: {list(df.columns)}")
        if not production_cols:
            errors.append(f"Solar file missing production column. Available columns: {list(df.columns)}")
        
        if not errors:
            datetime_col = datetime_cols[0]
            production_col = production_cols[0]
            
            if df[datetime_col].isnull().any():
                warnings.append("Some datetime values are missing in solar data")
            if df[production_col].isnull().any():
                warnings.append("Some production values are missing")
            
            try:
                numeric_values = pd.to_numeric(df[production_col], errors='coerce')
                if (numeric_values < 0).any():
                    warnings.append("Negative production values detected")
            except:
                warnings.append("Production column contains non-numeric values")
        
        return errors, warnings, datetime_cols[0] if datetime_cols else None, production_cols[0] if production_cols else None

def clean_solar_data(df, datetime_col, production_col):
    """Clean and prepare the solar data - handle both datetime and separated date columns"""
    df.columns = df.columns.str.strip()
    
    # Check if datetime_col is actually a tuple (Month, Day, Hour columns)
    if isinstance(datetime_col, tuple):
        month_col, day_col, hour_col = datetime_col
        
        # Handle month column - could be text or numbers
        if month_col in df.columns:
            month_mapping = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            df['Month_Numeric'] = df[month_col].map(month_mapping)
            if df['Month_Numeric'].isna().all():
                df['Month_Numeric'] = pd.to_numeric(df[month_col], errors='coerce')
        else:
            return None
        
        # Handle day column
        if day_col in df.columns:
            df['Day_Numeric'] = pd.to_numeric(df[day_col], errors='coerce')
        else:
            return None
        
        # Handle hour column - extract hour number from formats like "00:00", "01:00"
        if hour_col in df.columns:
            def extract_hour(hour_str):
                if pd.isna(hour_str):
                    return 0
                hour_str = str(hour_str).strip()
                if ':' in hour_str:
                    return int(hour_str.split(':')[0])
                else:
                    try:
                        return int(float(hour_str))
                    except:
                        return 0
            
            df['Hour_Numeric'] = df[hour_col].apply(extract_hour)
        else:
            return None
        
        # Create a generic datetime using 2024 (to match usage data year)
        try:
            df['DateTime'] = pd.to_datetime(
                df[['Month_Numeric', 'Day_Numeric']].rename(columns={'Month_Numeric': 'month', 'Day_Numeric': 'day'})
                .assign(year=2024, hour=df['Hour_Numeric'], minute=0, second=0)
            )
        except Exception as e:
            return None
        
    else:
        # Original datetime column handling
        df['DateTime'] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Handle production column
    if production_col in df.columns:
        df['Solar Production'] = pd.to_numeric(df[production_col], errors='coerce')
    else:
        return None
    
    # Remove rows with invalid data
    df = df.dropna(subset=['DateTime', 'Solar Production'])
    
    return df[['DateTime', 'Solar Production']].sort_values('DateTime')

def align_datasets(usage_df, solar_df):
    """Pure exact overlay: Match usage and solar data by exact date and hour only"""
    
    # Rename columns for consistency
    usage_aligned = usage_df.rename(columns={'Start Datetime': 'DateTime', 'Net Usage': 'Usage'})
    
    # Extract date and hour components for exact matching
    usage_aligned['Date'] = usage_aligned['DateTime'].dt.date
    usage_aligned['Hour'] = usage_aligned['DateTime'].dt.hour
    usage_aligned['Month'] = usage_aligned['DateTime'].dt.month
    usage_aligned['Day'] = usage_aligned['DateTime'].dt.day
    
    solar_df['Date'] = solar_df['DateTime'].dt.date  
    solar_df['Hour'] = solar_df['DateTime'].dt.hour
    solar_df['Month'] = solar_df['DateTime'].dt.month
    solar_df['Day'] = solar_df['DateTime'].dt.day
    
    # Try month/day/hour matching
    usage_aligned['MDH_Key'] = usage_aligned['Month'].astype(str) + '_' + usage_aligned['Day'].astype(str) + '_' + usage_aligned['Hour'].astype(str)
    solar_df['MDH_Key'] = solar_df['Month'].astype(str) + '_' + solar_df['Day'].astype(str) + '_' + solar_df['Hour'].astype(str)
    
    combined_df = pd.merge(
        usage_aligned,
        solar_df[['MDH_Key', 'Solar Production']],
        on='MDH_Key',
        how='left'
    )
    
    # Check results
    exact_matches = len(combined_df[combined_df['Solar Production'].notna()])
    total_records = len(combined_df)
    
    if exact_matches == 0:
        return None
    
    # Fill missing solar values with 0
    combined_df['Solar Production'] = combined_df['Solar Production'].fillna(0)
    
    # Clean up temporary columns
    combined_df = combined_df.drop(['MDH_Key'], axis=1)
    
    return combined_df

def calculate_energy_flows(df, battery_capacity, depth_of_discharge):
    """Calculate hourly energy flows with battery storage"""
    usable_capacity = battery_capacity * (1 - depth_of_discharge / 100)
    
    # Initialize new columns
    df['Solar Direct Use'] = 0.0
    df['Battery Charge'] = 0.0
    df['Battery Discharge'] = 0.0
    df['Battery SOC'] = 0.0
    df['Grid Draw'] = 0.0
    df['Grid Export'] = 0.0
    
    # Initialize battery state
    current_soc = 0.0  # Start with empty battery
    
    for i in range(len(df)):
        usage = df.iloc[i]['Usage']
        solar = df.iloc[i]['Solar Production']
        
        # Step 1: Solar Direct Use
        solar_direct = min(solar, usage)
        df.iloc[i, df.columns.get_loc('Solar Direct Use')] = solar_direct
        
        remaining_usage = usage - solar_direct
        excess_solar = solar - solar_direct
        
        # Step 2: Battery operations
        if excess_solar > 0:
            # Charge battery with excess solar
            available_battery_space = usable_capacity - current_soc
            battery_charge = min(excess_solar, available_battery_space)
            df.iloc[i, df.columns.get_loc('Battery Charge')] = battery_charge
            current_soc += battery_charge
            
            # Any remaining excess goes to grid export
            remaining_excess = excess_solar - battery_charge
            df.iloc[i, df.columns.get_loc('Grid Export')] = remaining_excess
            
        elif remaining_usage > 0:
            # Discharge battery to meet remaining usage
            battery_discharge = min(remaining_usage, current_soc)
            df.iloc[i, df.columns.get_loc('Battery Discharge')] = battery_discharge
            current_soc -= battery_discharge
            remaining_usage -= battery_discharge
            
            # Any remaining usage comes from grid
            df.iloc[i, df.columns.get_loc('Grid Draw')] = remaining_usage
        
        # Update SOC
        df.iloc[i, df.columns.get_loc('Battery SOC')] = current_soc
    
    return df

def generate_battery_insights(df, battery_capacity, depth_of_discharge):
    """Generate battery-specific insights"""
    insights = []
    usable_capacity = battery_capacity * (1 - depth_of_discharge / 100)
    
    # Grid independence analysis
    zero_grid_hours = len(df[df['Grid Draw'] == 0])
    total_hours = len(df)
    independence_pct = (zero_grid_hours / total_hours) * 100
    insights.append(f"🔋 **Grid Independence**: Achieved {independence_pct:.1f}% of the time ({zero_grid_hours:,} hours)")
    
    # Solar coverage
    total_usage = df['Usage'].sum()
    total_solar_direct = df['Solar Direct Use'].sum()
    solar_coverage = (total_solar_direct / total_usage) * 100
    insights.append(f"☀️ **Solar Coverage**: {solar_coverage:.1f}% of usage met directly by solar")
    
    # Grid consumption reduction
    total_grid_draw = df['Grid Draw'].sum()
    grid_reduction = ((total_usage - total_grid_draw) / total_usage) * 100
    insights.append(f"📉 **Grid Reduction**: {grid_reduction:.1f}% reduction in grid consumption with battery storage")
    
    # Battery backup time
    avg_hourly_usage = df['Usage'].mean()
    backup_hours = usable_capacity / avg_hourly_usage if avg_hourly_usage > 0 else 0
    insights.append(f"🏠 **Backup Power**: Battery provides approximately {backup_hours:.1f} hours of backup power")
    
    return insights

def create_monthly_energy_flow_chart(df, selected_month):
    """Create stacked area chart showing energy source mix for selected month"""
    month_df = df[df['DateTime'].dt.month == selected_month]
    
    if len(month_df) == 0:
        return None
    
    daily_data = month_df.groupby('Date').agg({
        'Solar Direct Use': 'sum',
        'Battery Discharge': 'sum',
        'Grid Draw': 'sum',
        'Usage': 'sum'
    }).reset_index()
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['Solar Direct Use'],
        fill='tonexty',
        mode='none',
        name='Solar Direct Use',
        fillcolor='rgba(255, 193, 7, 0.7)',
        line=dict(color='rgba(255, 193, 7, 1)')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['Solar Direct Use'] + daily_data['Battery Discharge'],
        fill='tonexty',
        mode='none',
        name='Battery Discharge',
        fillcolor='rgba(40, 167, 69, 0.7)',
        line=dict(color='rgba(40, 167, 69, 1)')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['Usage'],
        fill='tonexty',
        mode='none',
        name='Grid Draw',
        fillcolor='rgba(220, 53, 69, 0.7)',
        line=dict(color='rgba(220, 53, 69, 1)')
    ))
    
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    fig.update_layout(
        title=f"{month_names[selected_month]} Energy Source Mix",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        height=500
    )
    
    return fig

def create_monthly_battery_cycles_chart(df, selected_month):
    """Create monthly battery charge/discharge chart"""
    month_df = df[df['DateTime'].dt.month == selected_month]
    
    if len(month_df) == 0:
        return None
    
    daily_cycles = month_df.groupby('Date').agg({
        'Battery Charge': 'sum',
        'Battery Discharge': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_cycles['Date'],
        y=daily_cycles['Battery Charge'],
        name='Daily Charge',
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=daily_cycles['Date'],
        y=-daily_cycles['Battery Discharge'],
        name='Daily Discharge',
        marker_color='red'
    ))
    
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    fig.update_layout(
        title=f"{month_names[selected_month]} Battery Charge/Discharge Cycles",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        height=400
    )
    return fig

def create_monthly_hourly_patterns(df, selected_month):
    """Create hourly patterns for selected month"""
    month_df = df[df['DateTime'].dt.month == selected_month]
    
    if len(month_df) == 0:
        return None
    
    hourly_pattern = month_df.groupby('Hour').agg({
        'Usage': 'mean',
        'Solar Production': 'mean',
        'Solar Direct Use': 'mean',
        'Battery Discharge': 'mean',
        'Grid Draw': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Usage'], name='Usage', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Solar Production'], name='Solar Production', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Solar Direct Use'], name='Solar Direct Use', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Battery Discharge'], name='Battery Discharge', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Grid Draw'], name='Grid Draw', line=dict(color='blue')))
    
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    fig.update_layout(
        title=f"{month_names[selected_month]} Average Hourly Energy Patterns",
        xaxis_title="Hour of Day",
        yaxis_title="Energy (kWh)",
        height=500
    )
    return fig

# ============= MAIN APP =============

st.markdown('<h1 class="main-header">⚡ PowerInsight</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Energy Analytics Platform</p>', unsafe_allow_html=True)

# Main mode selection tabs
main_tab1, main_tab2 = st.tabs(["📊 Usage Analysis", "🔋 Solar + Storage Analysis"])

# ============= USAGE ANALYSIS TAB =============
with main_tab1:
    # Sidebar for Usage Analysis
    with st.sidebar:
        st.header("📁 Usage Data Upload")
        usage_file = st.file_uploader(
            "Upload usage CSV file", 
            type=["csv"],
            help="CSV with 'Start Datetime' and 'Net Usage' columns",
            key="usage_file_main"
        )
        
        if usage_file:
            st.success("✅ Usage file uploaded successfully!")
            
            st.header("⚙️ Analysis Settings")
            
            # Day/night configuration
            st.subheader("Time Period Settings")
            
            # Day start time
            day_hour = st.selectbox("Day starts at:", 
                                   options=list(range(1, 13)), 
                                   index=5,  # Default to 6 AM
                                   key="usage_day_hour")
            day_period = st.selectbox("", options=["AM", "PM"], key="usage_day_period")
            day_start_24 = convert_12_to_24_hour(day_hour, day_period)
            
            # Night start time  
            night_hour = st.selectbox("Night starts at:", 
                                     options=list(range(1, 13)), 
                                     index=5,  # Default to 6 PM
                                     key="usage_night_hour")
            night_period = st.selectbox("", options=["AM", "PM"], 
                                       index=1,  # Default to PM
                                       key="usage_night_period")
            night_start_24 = convert_12_to_24_hour(night_hour, night_period)
            
            st.info(f"Day: {convert_24_to_12_hour(day_start_24)} - {convert_24_to_12_hour(night_start_24)}")
            st.info(f"Night: {convert_24_to_12_hour(night_start_24)} - {convert_24_to_12_hour(day_start_24)}")

    if usage_file is not None:
        try:
            # Load and validate usage data
            df = pd.read_csv(usage_file)
            
            with st.spinner("🔍 Validating data..."):
                errors, warnings = validate_usage_data(df)
            
            if errors:
                st.error("❌ **Data Validation Errors:**")
                for error in errors:
                    st.error(f"• {error}")
                st.stop()
            
            if warnings:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ **Data Quality Warnings:**")
                for warning in warnings:
                    st.warning(f"• {warning}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean data
            with st.spinner("🧹 Cleaning and processing data..."):
                df = clean_usage_data(df)
                
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ Successfully processed {len(df):,} data points from {df['Start Datetime'].min().strftime('%Y-%m-%d')} to {df['Start Datetime'].max().strftime('%Y-%m-%d')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Monthly Analysis
            months = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"]
            monthly_analysis = []
            
            for month in range(1, 13):
                month_df = df[df['Month'] == month]
                
                if len(month_df) == 0:
                    continue
                    
                total_usage = month_df['Net Usage'].sum()
                num_days_in_month = len(month_df['Day'].unique())
                avg_24hr_usage = total_usage / num_days_in_month if num_days_in_month > 0 else 0
                
                # Use customizable day/night hours
                daytime_usage = month_df[(month_df['Hour'] >= day_start_24) & (month_df['Hour'] < night_start_24)]['Net Usage'].sum()
                nighttime_usage = total_usage - daytime_usage
                
                avg_daytime_usage = daytime_usage / num_days_in_month if num_days_in_month > 0 else 0
                avg_nighttime_usage = nighttime_usage / num_days_in_month if num_days_in_month > 0 else 0
                
                # Peak hour analysis
                hourly_avg = month_df.groupby('Hour')['Net Usage'].mean()
                peak_hour = hourly_avg.idxmax() if len(hourly_avg) > 0 else 0
                peak_usage = hourly_avg.max() if len(hourly_avg) > 0 else 0
                
                monthly_data = {
                    'Month': months[month-1],
                    'Total Usage': total_usage,
                    'Average 24 Hour Usage': avg_24hr_usage,
                    'Average Daytime Usage': avg_daytime_usage,
                    'Average Nighttime Usage': avg_nighttime_usage,
                    'Peak Hour': peak_hour,
                    'Peak Hour Usage': peak_usage
                }
                
                monthly_analysis.append(monthly_data)
            
            # Generate insights
            insights = generate_usage_insights(df, monthly_analysis)
            
            # Display insights
            st.header("🧠 Smart Insights")
            insight_cols = st.columns(2)
            for i, insight in enumerate(insights):
                with insight_cols[i % 2]:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Enhanced visualizations
            st.header("📈 Advanced Analytics")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Monthly Overview", "📅 Daily Patterns", "🌿 Seasonal Analysis"])
            
            with tab1:
                if monthly_analysis:
                    monthly_df = pd.DataFrame(monthly_analysis)
                    
                    # Monthly usage bar chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Monthly Total Usage', 'Day vs Night Usage Comparison'),
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                    )
                    
                    # Total usage bar chart
                    fig.add_trace(
                        go.Bar(
                            x=monthly_df['Month'],
                            y=monthly_df['Total Usage'],
                            name='Total Usage',
                            marker_color='#667eea',
                            hovertemplate='<b>%{x}</b><br>Total Usage: %{y:.1f} kWh<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Day vs Night comparison
                    fig.add_trace(
                        go.Bar(
                            x=monthly_df['Month'],
                            y=monthly_df['Average Daytime Usage'],
                            name='Daytime',
                            marker_color='#ffa726',
                            hovertemplate='<b>%{x}</b><br>Daytime: %{y:.1f} kWh<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=monthly_df['Month'],
                            y=monthly_df['Average Nighttime Usage'],
                            name='Nighttime',
                            marker_color='#42a5f5',
                            hovertemplate='<b>%{x}</b><br>Nighttime: %{y:.1f} kWh<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=800, showlegend=True, title_text="Monthly Usage Analysis")
                    fig.update_xaxes(title_text="Month", row=2, col=1)
                    fig.update_yaxes(title_text="Usage (kWh)", row=1, col=1)
                    fig.update_yaxes(title_text="Average Daily Usage (kWh)", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly Summary
                    st.subheader("📊 Monthly Summary")
                    st.dataframe(monthly_df.style.format({
                        'Total Usage': '{:.1f} kWh',
                        'Average 24 Hour Usage': '{:.2f} kWh',
                        'Average Daytime Usage': '{:.2f} kWh',
                        'Average Nighttime Usage': '{:.2f} kWh',
                        'Peak Hour Usage': '{:.2f} kWh'
                    }), use_container_width=True)
                    
                    # Annual Summary
                    st.subheader("📊 Annual Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    yearly_total = sum([x['Total Usage'] for x in monthly_analysis])
                    yearly_24hr = sum([x['Average 24 Hour Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                    yearly_day = sum([x['Average Daytime Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                    yearly_night = sum([x['Average Nighttime Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                    
                    with col1:
                        st.metric("Total Annual Usage", f"{yearly_total:.0f} kWh")
                    with col2:
                        st.metric("Avg Daily Usage", f"{yearly_24hr:.1f} kWh")
                    with col3:
                        st.metric("Avg Daytime Usage", f"{yearly_day:.1f} kWh")
                    with col4:
                        st.metric("Avg Nighttime Usage", f"{yearly_night:.1f} kWh")
            
            with tab2:
                # Daily usage trends
                daily_usage = df.groupby('Date')['Net Usage'].sum().reset_index()
                daily_usage['Date'] = pd.to_datetime(daily_usage['Date'])
                
                fig = px.line(
                    daily_usage, 
                    x='Date', 
                    y='Net Usage',
                    title="📅 Daily Usage Trends Over Time"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly patterns
                weekly_pattern = df.groupby('DayOfWeek')['Net Usage'].mean().reindex([
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                ])
                
                fig = go.Figure(data=go.Bar(
                    x=weekly_pattern.index,
                    y=weekly_pattern.values,
                    marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
                ))
                fig.update_layout(title="📊 Average Usage by Day of Week", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_seasonal_chart(df), use_container_width=True)
                
                # Seasonal summary
                seasonal_summary = df.groupby('Season')['Net Usage'].agg(['sum', 'mean']).round(2)
                seasonal_summary.columns = ['Total Usage (kWh)', 'Average Usage (kWh)']
                st.subheader("Seasonal Usage Summary")
                st.dataframe(seasonal_summary)
            
            # Data export option
            st.header("📥 Export Results")
            if st.button("📊 Download Analysis Results", key="usage_export"):
                # Create downloadable CSV
                export_df = pd.DataFrame(monthly_analysis)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"usage_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="usage_download"
                )
        
        except Exception as e:
            st.error(f"❌ **Error processing file**: {str(e)}")
            st.info("💡 **Tip**: Make sure your CSV file has 'Start Datetime' and 'Net Usage' columns with valid data.")
    
    else:
        # Landing page for usage analysis
        st.markdown("""
        ## 🚀 Welcome to Usage Analysis!
        
        Upload your electrical usage CSV file to get started with advanced analytics including:
        
        ### 📊 **Comprehensive Analysis**
        - Monthly, seasonal, and yearly usage patterns
        - Peak demand identification
        - Day vs night usage comparison with customizable time periods
        
        ### 📈 **Advanced Visualizations**
        - Seasonal trend analysis and daily usage graphs
        - Interactive monthly usage charts
        - Weekly pattern analysis
        
        ### 🧠 **Smart Insights**
        - Automated pattern recognition and recommendations
        - Comparative analysis across different time periods
        - Usage trend identification
        
        ### 📁 **File Requirements**
        Your CSV file should contain these columns:
        - `Start Datetime`: Timestamp of the reading
        - `Net Usage`: Energy usage in kWh
        
        ---
        
        **Ready to analyze your energy usage? Upload your file using the sidebar! 👈**
        """)
        
        # Sample data format
        with st.expander("📋 Sample Data Format"):
            sample_data = pd.DataFrame({
                'Start Datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
                'Net Usage': [1.25, 0.85, 0.65]
            })
            st.dataframe(sample_data)

# ============= SOLAR + STORAGE ANALYSIS TAB =============
with main_tab2:
    # Sidebar for Solar + Storage Analysis
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Usage data upload for solar analysis
        st.subheader("Electrical Usage Data")
        usage_file_solar = st.file_uploader(
            "Upload usage CSV file", 
            type=["csv"],
            help="CSV with 'Start Datetime' and 'Net Usage' columns",
            key="usage_file_solar"
        )
        
        # Solar data upload
        st.subheader("Solar Production Data")
        solar_file = st.file_uploader(
            "Upload solar production CSV file", 
            type=["csv"],
            help="CSV with datetime and energy production (kWh) columns",
            key="solar_file"
        )
        
        if usage_file_solar and solar_file:
            st.success("✅ Both files uploaded successfully!")
            
            st.header("🔋 Battery Configuration")
            battery_capacity = st.number_input(
                "Battery Total Capacity (kWh)",
                min_value=1.0,
                max_value=200.0,
                value=15.0,  # Changed default to 15 kWh
                step=0.5,
                help="Total installed battery capacity"
            )
            
            depth_of_discharge = st.slider(
                "Depth of Discharge (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Percentage of battery capacity reserved (not usable)"
            )
            
            usable_capacity = battery_capacity * (1 - depth_of_discharge / 100)
            st.markdown(f'<div class="battery-box"><strong>Usable Capacity:</strong> {usable_capacity:.1f} kWh</div>', unsafe_allow_html=True)
            
            st.header("⚙️ Analysis Settings")
            
            # Day/night configuration
            st.subheader("Time Period Settings")
            
            # Day start time
            solar_day_hour = st.selectbox("Day starts at:", 
                                   options=list(range(1, 13)), 
                                   index=5,  # Default to 6 AM
                                   key="solar_day_hour")
            solar_day_period = st.selectbox("", options=["AM", "PM"], key="solar_day_period")
            solar_day_start_24 = convert_12_to_24_hour(solar_day_hour, solar_day_period)
            
            # Night start time  
            solar_night_hour = st.selectbox("Night starts at:", 
                                     options=list(range(1, 13)), 
                                     index=5,  # Default to 6 PM
                                     key="solar_night_hour")
            solar_night_period = st.selectbox("", options=["AM", "PM"], 
                                       index=1,  # Default to PM
                                       key="solar_night_period")
            solar_night_start_24 = convert_12_to_24_hour(solar_night_hour, solar_night_period)
            
            st.info(f"Day: {convert_24_to_12_hour(solar_day_start_24)} - {convert_24_to_12_hour(solar_night_start_24)}")
            st.info(f"Night: {convert_24_to_12_hour(solar_night_start_24)} - {convert_24_to_12_hour(solar_day_start_24)}")

    # Main content for Solar + Storage Analysis
    if usage_file_solar is not None and solar_file is not None:
        try:
            # Load and validate usage data
            usage_df = pd.read_csv(usage_file_solar)
            with st.spinner("🔍 Validating usage data..."):
                usage_errors, usage_warnings = validate_usage_data(usage_df)
            
            # Load and validate solar data
            solar_df = pd.read_csv(solar_file)
            with st.spinner("🔍 Validating solar data..."):
                solar_errors, solar_warnings, datetime_col, production_col = validate_solar_data(solar_df)
            
            # Check for errors
            all_errors = usage_errors + solar_errors
            if all_errors:
                st.error("❌ **Data Validation Errors:**")
                for error in all_errors:
                    st.error(f"• {error}")
                st.stop()
            
            # Display warnings
            all_warnings = usage_warnings + solar_warnings
            if all_warnings:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ **Data Quality Warnings:**")
                for warning in all_warnings:
                    st.warning(f"• {warning}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean data
            with st.spinner("🧹 Processing data..."):
                usage_clean = clean_usage_data(usage_df)
                solar_clean = clean_solar_data(solar_df, datetime_col, production_col)
                
                if solar_clean is None:
                    st.error("❌ Error processing solar data")
                    st.stop()
                
                # Align datasets
                combined_df = align_datasets(usage_clean, solar_clean)
                
                if combined_df is None:
                    st.error("❌ No matching data found between usage and solar files")
                    st.stop()
                
                # Calculate energy flows
                result_df = calculate_energy_flows(combined_df, battery_capacity, depth_of_discharge)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ Successfully processed {len(result_df):,} data points from {result_df['DateTime'].min().strftime('%Y-%m-%d')} to {result_df['DateTime'].max().strftime('%Y-%m-%d')}")
            st.info("🔄 **Pattern Matching**: Solar production patterns have been overlaid onto your historical usage timeline to simulate system performance.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate insights
            insights = generate_battery_insights(result_df, battery_capacity, depth_of_discharge)
            
            # Display insights
            st.header("🧠 Smart Insights")
            insight_cols = st.columns(2)
            for i, insight in enumerate(insights):
                with insight_cols[i % 2]:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Key Metrics Dashboard
            st.header("📊 Key Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_usage = result_df['Usage'].sum()
                st.metric("Total Usage", f"{total_usage:.0f} kWh")
            
            with col2:
                total_solar = result_df['Solar Production'].sum()
                st.metric("Total Solar Production", f"{total_solar:.0f} kWh")
            
            with col3:
                total_grid = result_df['Grid Draw'].sum()
                st.metric("Grid Consumption", f"{total_grid:.0f} kWh")
            
            with col4:
                grid_independence = (1 - total_grid / total_usage) * 100
                st.metric("Grid Independence", f"{grid_independence:.1f}%")
            
            # Advanced visualizations - Solar + Storage Analysis
            st.header("📈 Advanced Analytics")
            
            # Create tabs for different views (Traditional Analysis first)
            tab1, tab2, tab3, tab4 = st.tabs([
                "📅 Traditional Analysis",
                "🔋 Battery Performance", 
                "⚡ Energy Flow Analysis", 
                "📊 System Comparison"
            ])
            
            with tab1:
                # Monthly analysis with January first
                months_order = ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"]
                
                monthly_data_list = []
                for month_num in range(1, 13):
                    month_df = result_df[result_df['DateTime'].dt.month == month_num]
                    if len(month_df) > 0:
                        monthly_data_list.append({
                            'Month': months_order[month_num - 1],
                            'Total Usage': month_df['Usage'].sum(),
                            'Solar Production': month_df['Solar Production'].sum(),
                            'Grid Consumption': month_df['Grid Draw'].sum()
                        })
                
                if monthly_data_list:
                    monthly_data_df = pd.DataFrame(monthly_data_list)
                    
                    # Monthly energy summary graph
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=monthly_data_df['Month'], y=monthly_data_df['Total Usage'], name='Total Usage'))
                    fig.add_trace(go.Bar(x=monthly_data_df['Month'], y=monthly_data_df['Solar Production'], name='Solar Production'))
                    fig.add_trace(go.Bar(x=monthly_data_df['Month'], y=monthly_data_df['Grid Consumption'], name='Grid Consumption'))
                    
                    fig.update_layout(
                        title="Monthly Energy Summary",
                        xaxis_title="Month",
                        yaxis_title="Energy (kWh)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly data table
                    st.subheader("Monthly Energy Summary Table")
                    st.dataframe(monthly_data_df.style.format({
                        'Total Usage': '{:.1f} kWh',
                        'Solar Production': '{:.1f} kWh',
                        'Grid Consumption': '{:.1f} kWh'
                    }), use_container_width=True)
                
                # Monthly breakdown of grid energy timing
                st.subheader("Monthly Grid Energy Usage Timing")
                
                # Grid energy timing analysis
                grid_timing_data = []
                for month_num in range(1, 13):
                    month_df = result_df[result_df['DateTime'].dt.month == month_num]
                    if len(month_df) > 0:
                        # Filter for grid draw only
                        grid_df = month_df[month_df['Grid Draw'] > 0]
                        
                        if len(grid_df) > 0:
                            # Calculate day vs night grid usage
                            day_grid = grid_df[(grid_df['DateTime'].dt.hour >= solar_day_start_24) & 
                                             (grid_df['DateTime'].dt.hour < solar_night_start_24)]['Grid Draw'].sum()
                            night_grid = grid_df['Grid Draw'].sum() - day_grid
                            
                            grid_timing_data.append({
                                'Month': months_order[month_num - 1],
                                'Daytime Grid Draw': day_grid,
                                'Nighttime Grid Draw': night_grid,
                                'Total Grid Draw': grid_df['Grid Draw'].sum()
                            })
                
                if grid_timing_data:
                    grid_timing_df = pd.DataFrame(grid_timing_data)
                    
                    # Grid timing chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=grid_timing_df['Month'], 
                        y=grid_timing_df['Daytime Grid Draw'], 
                        name='Daytime Grid Draw',
                        marker_color='orange'
                    ))
                    fig.add_trace(go.Bar(
                        x=grid_timing_df['Month'], 
                        y=grid_timing_df['Nighttime Grid Draw'], 
                        name='Nighttime Grid Draw',
                        marker_color='blue'
                    ))
                    
                    fig.update_layout(
                        title=f"Monthly Grid Energy Usage - Day ({convert_24_to_12_hour(solar_day_start_24)}-{convert_24_to_12_hour(solar_night_start_24)}) vs Night",
                        xaxis_title="Month",
                        yaxis_title="Grid Energy (kWh)",
                        height=400,
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Grid timing table
                    st.subheader("Grid Energy Timing Table")
                    st.dataframe(grid_timing_df.style.format({
                        'Daytime Grid Draw': '{:.1f} kWh',
                        'Nighttime Grid Draw': '{:.1f} kWh',
                        'Total Grid Draw': '{:.1f} kWh'
                    }), use_container_width=True)
                
                # Seasonal patterns
                result_df['Season'] = result_df['DateTime'].dt.month.map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                seasonal_data = result_df.groupby(['Season', 'Hour']).agg({
                    'Usage': 'mean',
                    'Solar Production': 'mean'
                }).reset_index()
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Usage Patterns', 'Solar Production Patterns'))
                
                for season in seasonal_data['Season'].unique():
                    season_data = seasonal_data[seasonal_data['Season'] == season]
                    fig.add_trace(
                        go.Scatter(x=season_data['Hour'], y=season_data['Usage'], 
                                 name=f'{season} Usage', mode='lines'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=season_data['Hour'], y=season_data['Solar Production'], 
                                 name=f'{season} Solar', mode='lines'),
                        row=1, col=2
                    )
                
                fig.update_layout(height=500, title_text="Seasonal Energy Patterns")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Month selector for battery performance
                available_months = sorted(result_df['DateTime'].dt.month.unique())
                month_names = ["", "January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December"]
                
                battery_month = st.selectbox(
                    "Select Month for Battery Analysis:",
                    options=available_months,
                    format_func=lambda x: month_names[x],
                    key="battery_month_selector"
                )
                
                # Monthly battery cycles chart
                fig_cycles = create_monthly_battery_cycles_chart(result_df, battery_month)
                if fig_cycles:
                    st.plotly_chart(fig_cycles, use_container_width=True)
                
                # Monthly battery utilization metrics
                month_df = result_df[result_df['DateTime'].dt.month == battery_month]
                if len(month_df) > 0:
                    st.subheader(f"{month_names[battery_month]} Battery Utilization Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        monthly_discharge = month_df.groupby('Date')['Battery Discharge'].sum().mean()
                        monthly_utilization = (monthly_discharge / usable_capacity) * 100
                        st.metric("Monthly Avg Daily Utilization", f"{monthly_utilization:.1f}%")
                    
                    with col2:
                        cycling_days = len(month_df[month_df['Battery Discharge'] > 0].groupby('Date'))
                        total_days = len(month_df['Date'].unique())
                        st.metric("Monthly Cycling Days", f"{cycling_days}/{total_days}")
            
            with tab3:
                # Month selector for energy flow
                flow_month = st.selectbox(
                    "Select Month for Energy Flow Analysis:",
                    options=available_months,
                    format_func=lambda x: month_names[x],
                    key="flow_month_selector"
                )
                
                st.subheader("Monthly Energy Source Mix")
                fig_flow = create_monthly_energy_flow_chart(result_df, flow_month)
                if fig_flow:
                    st.plotly_chart(fig_flow, use_container_width=True)
                
                # Monthly hourly patterns
                st.subheader("Monthly Average Hourly Energy Patterns")
                pattern_month = st.selectbox(
                    "Select Month for Hourly Patterns:",
                    options=available_months,
                    format_func=lambda x: month_names[x],
                    key="pattern_month_selector"
                )
                
                fig_patterns = create_monthly_hourly_patterns(result_df, pattern_month)
                if fig_patterns:
                    st.plotly_chart(fig_patterns, use_container_width=True)
            
            with tab4:
                st.subheader("System Performance Comparison")
                
                # Calculate scenario without battery
                no_battery_grid = np.maximum(0, result_df['Usage'] - result_df['Solar Direct Use'])
                
                comparison_data = pd.DataFrame({
                    'Scenario': ['With Battery Storage', 'Without Battery Storage'],
                    'Grid Consumption (kWh)': [result_df['Grid Draw'].sum(), no_battery_grid.sum()]
                })
                
                fig = go.Figure(data=go.Bar(
                    x=comparison_data['Scenario'], 
                    y=comparison_data['Grid Consumption (kWh)'],
                    marker_color=['green', 'red']
                ))
                fig.update_layout(
                    title='Grid Consumption Comparison',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Savings summary
                grid_savings = no_battery_grid.sum() - result_df['Grid Draw'].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Grid Consumption Reduction", f"{grid_savings:.0f} kWh", f"{(grid_savings/no_battery_grid.sum()*100):.1f}%")
                with col2:
                    solar_with_battery = (result_df['Solar Direct Use'].sum() + result_df['Battery Charge'].sum()) / result_df['Solar Production'].sum() * 100
                    solar_without_battery = result_df['Solar Direct Use'].sum() / result_df['Solar Production'].sum() * 100
                    solar_improvement = solar_with_battery - solar_without_battery
                    st.metric("Solar Utilization Improvement", f"{solar_improvement:.1f}%")
            
            # Data export options
            st.header("📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Complete Analysis", key="solar_complete_export"):
                    export_df = result_df.copy()
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download Complete Dataset CSV",
                        data=csv,
                        file_name=f"solar_battery_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="solar_complete_download"
                    )
            
            with col2:
                if st.button("📈 Download Summary Report", key="solar_summary_export"):
                    # Create summary report
                    summary_data = {
                        'Metric': [
                            'Total Usage (kWh)',
                            'Total Solar Production (kWh)',
                            'Grid Consumption (kWh)',
                            'Grid Independence (%)',
                            'Solar Coverage (%)',
                            'Total Battery Cycles (kWh)',
                            'Grid Consumption Reduction (kWh)'
                        ],
                        'Value': [
                            result_df['Usage'].sum(),
                            result_df['Solar Production'].sum(),
                            result_df['Grid Draw'].sum(),
                            (1 - result_df['Grid Draw'].sum() / result_df['Usage'].sum()) * 100,
                            (result_df['Solar Direct Use'].sum() / result_df['Usage'].sum()) * 100,
                            result_df['Battery Discharge'].sum(),
                            np.maximum(0, result_df['Usage'] - result_df['Solar Direct Use']).sum() - result_df['Grid Draw'].sum()
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary Report CSV",
                        data=summary_csv,
                        file_name=f"solar_battery_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="solar_summary_download"
                    )
        
        except Exception as e:
            st.error(f"❌ **Error processing files**: {str(e)}")
            st.info("💡 **Tips**: Make sure your CSV files have the correct columns and compatible datetime formats.")

    elif usage_file_solar is not None:
        st.info("📋 **Next Step**: Upload your solar production data to enable complete solar + battery analysis")
        
        # Show preview of usage data
        try:
            usage_df = pd.read_csv(usage_file_solar)
            st.subheader("Usage Data Preview")
            st.dataframe(usage_df.head())
        except Exception as e:
            st.error(f"Error reading usage file: {str(e)}")

    elif solar_file is not None:
        st.info("📋 **Next Step**: Upload your electrical usage data to enable complete solar + battery analysis")
        
        # Show preview of solar data
        try:
            solar_df = pd.read_csv(solar_file)
            st.subheader("Solar Data Preview")
            st.dataframe(solar_df.head())
        except Exception as e:
            st.error(f"Error reading solar file: {str(e)}")

    else:
        # Landing page when no files are uploaded for solar + storage
        st.markdown("""
        ## 🚀 Welcome to Solar + Storage Analysis!
        
        Upload both your electrical usage and solar production CSV files to unlock comprehensive solar + battery storage analysis.
        
        ### 🔋 **Battery Storage Features**
        - **Energy Flow Analysis**: Track solar direct use, battery charging/discharging, and grid consumption
        - **Battery Performance**: Monitor utilization and cycling patterns on a monthly basis
        - **Grid Independence**: Measure hours of complete energy self-sufficiency
        - **System Optimization**: Compare performance with and without battery storage
        
        ### 📊 **Enhanced Analytics Include**
        - **Monthly Energy Flow**: Month-by-month breakdown of where your power comes from
        - **Battery Cycling**: Monthly charge/discharge patterns
        - **Grid Consumption Analysis**: Detailed timing of when grid power is used
        - **Traditional Analysis**: Comprehensive monthly and seasonal breakdowns
        
        ### ⚡ **Smart Energy Insights**
        - Battery utilization patterns by month
        - Grid consumption reduction with battery storage  
        - Complete grid independence analysis
        - Solar coverage: percentage of usage met directly by solar
        
        ### 📁 **File Requirements**
        
        **Electrical Usage CSV:**
        - `Start Datetime`: Timestamp of the reading
        - `Net Usage`: Energy usage in kWh
        
        **Solar Production CSV:**
        - Datetime column (various formats accepted)
        - Production/Generation column in kWh
        
        ### 🔧 **Battery Configuration**
        - Set your battery capacity (default: 15 kWh) and depth of discharge limits
        - Calculate usable capacity automatically
        - Model realistic charging/discharging behavior
        
        ---
        
        **Ready to optimize your solar + storage system? Upload both files using the sidebar! 👈**
        """)
        
        # Sample data formats
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📋 Sample Usage Data Format"):
                sample_usage = pd.DataFrame({
                    'Start Datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
                    'Net Usage': [1.25, 0.85, 0.65]
                })
                st.dataframe(sample_usage)
        
        with col2:
            with st.expander("☀️ Sample Solar Data Format"):
                sample_solar = pd.DataFrame({
                    'DateTime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
                    'Solar Production': [0.0, 0.0, 0.0]
                })
                st.dataframe(sample_solar)
        
        # Feature highlights
        st.markdown("""
        ### 🌟 **Key Features**
        
        #### 📅 Traditional Analysis (Enhanced)
        - **Monthly Energy Summary**: Complete breakdown starting with January
        - **Grid Usage Timing**: Day vs night grid consumption with customizable time periods
        - **Seasonal Patterns**: Understanding when energy flows occur throughout the year
        
        #### 🔋 Battery Performance Analysis (Monthly Focus)
        - **Monthly Battery Cycles**: Charge/discharge patterns by selected month
        - **Monthly Utilization**: Battery usage efficiency month by month
        - **Cycling Analysis**: Track active battery usage days per month
        
        #### ⚡ Energy Flow Visualization (Monthly Views)
        - **Monthly Energy Mix**: See energy source breakdown for any selected month
        - **Monthly Hourly Patterns**: Understand hourly energy flows by month
        - **Interactive Month Selection**: Toggle between months for detailed analysis
        
        #### 📈 System Comparison
        - **With vs Without Battery**: Compare grid consumption scenarios
        - **Grid Independence**: Track complete energy self-sufficiency
        - **Performance Metrics**: Efficiency calculations and recommendations
        """)

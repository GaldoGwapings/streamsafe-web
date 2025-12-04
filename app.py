import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import json
import os

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Stream Safe",
    page_icon="logo.svg",  
    layout="centered" 
)

DB_FILE = "water_database.json"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* IMPORT LEXEND FONT */
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

    /* 1. Global Font Application */
    html, body, [class*="css"] {
        font-family: 'Lexend', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #E3F2FD 0%, #F1F8E9 100%);
        background-attachment: fixed;
    }

    /* Reduce Top Padding to avoid empty whitespace */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* 2. Soft, Rounded Containers (Compact Padding) */
    [data-testid="stForm"], [data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background-color: white;
        border-radius: 15px;
        padding: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #E1E4E8;
    }
    
    /* 3. Rounded Inputs */
    .stTextInput input, .stNumberInput input, .stDateInput input {
        border-radius: 10px !important;
        border: 1px solid #CFD8DC;
        padding: 8px;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid #CFD8DC;
    }
    
    /* 4. Center Title */
    h1 {
        text-align: center;
        color: #0277BD;
        padding: 0px !important;
        margin: 0px 0px 10px 0px !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* 5. Metric Cards (Compact) */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #eee;
        padding: 10px 0px; 
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }

    [data-testid="stMetricLabel"] {
        justify-content: center !important;
        width: 100%;
        display: flex;
        font-size: 0.8rem !important;
        color: #666;
    }

    [data-testid="stMetricValue"] {
        justify-content: center !important;
        width: 100%;
        display: flex;
        font-weight: 700;
        color: #222;
        font-size: 1.5rem !important;
    }
    
    /* 6. Buttons */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid transparent;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #555;
        width: 100%;
        font-family: 'Lexend', sans-serif;
    }
    div.stButton > button:hover {
        border-color: #0277BD;
        color: #0277BD;
        background-color: #F1F8E9;
    }
    
    /* Action Buttons inside Forms */
    [data-testid="stForm"] div.stButton > button {
        background-color: #0277BD !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Image Centering Helper */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Locations
LOCATIONS = {
    "Pulangi River Station": [8.1575, 125.1313],
    "Sawaga River Point": [8.1628, 125.1200],
    "Malaybalay Reservoir": [8.1489, 125.1280],
    "Barangay Casisang Well": [8.1712, 125.1150],
    "Sumpong Water Station": [8.1590, 125.1250],
    "Bangcud Creek": [8.1200, 125.1300]
}

# --- 2. DATABASE FUNCTIONS ---
def load_data():
    if not os.path.exists(DB_FILE):
        default_data = [
            {"Location": "Pulangi River Station", "lat": 8.1575, "lon": 125.1313,
             "Risk": "Low Risk", "Date": "2025-12-01 08:30 AM", "Confidence": "98%", "pH": 7.2, "TSS": 15},
            {"Location": "Barangay Casisang Well", "lat": 8.1712, "lon": 125.1150,
             "Risk": "High Risk", "Date": "2025-12-02 02:15 PM", "Confidence": "92%", "pH": 5.5, "TSS": 120}
        ]
        with open(DB_FILE, 'w') as f:
            json.dump(default_data, f)
        return default_data
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_new_entry(entry):
    current_data = load_data()
    current_data.append(entry)
    with open(DB_FILE, 'w') as f:
        json.dump(current_data, f)

if 'user_session' not in st.session_state:
    st.session_state.user_session = None

if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = 'Sample List'

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_pipeline():
    try:
        model_pipeline = joblib.load('streamsafe_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model_pipeline, encoder
    except FileNotFoundError:
        return None, None

model, encoder = load_pipeline()

# --- 4. FUNCTIONS ---

def login():
    st.markdown('<h1>Stream Safe</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#546E7A; margin-bottom: 20px; font-size: 0.9rem;">Water Quality Monitoring System</p>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="user@safestream.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # UPDATED: Replaced use_container_width=True with width="stretch"
        submitted = st.form_submit_button("Sign In", width="stretch")
        
        if submitted:
            if email == "admin@safestream.com" and password == "Admin123":
                st.session_state.user_session = "admin"
                st.session_state.nav_selection = "Sample List"
                st.rerun()
            elif email == "user@safestream.com" and password == "User123":
                st.session_state.user_session = "user"
                st.rerun()
            else:
                st.error("Invalid credentials.")

def logout():
    st.session_state.user_session = None
    st.rerun()

def get_risk_color(risk):
    if "High" in risk: return "#D32F2F"
    if "Moderate" in risk: return "#F57C00"
    return "#388E3C"

# --- 5. ADMIN VIEW ---
def admin_view():
    data_log = load_data()

    st.markdown("<h1>Admin</h1>", unsafe_allow_html=True)

    # Nav Buttons (Compact)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        # UPDATED: Replaced use_container_width=True with width="stretch"
        if st.button("📋 Sample List", width="stretch"):
            st.session_state.nav_selection = "Sample List"
            st.rerun()
    with c2:
        # UPDATED: Replaced use_container_width=True with width="stretch"
        if st.button("➕ Add Sample", width="stretch"):
            st.session_state.nav_selection = "Add Sample Entry"
            st.rerun()

    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    if st.session_state.nav_selection == "Sample List":
        # Stats
        total = len(data_log)
        high_risk_count = sum(1 for x in data_log if "High" in x['Risk'])
        safe_count = sum(1 for x in data_log if "Low" in x['Risk'])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Risk", high_risk_count)
        c3.metric("Safe", safe_count)
        
        st.subheader("Map Overview")
        map_data = []
        for loc_name, coords in LOCATIONS.items():
            map_data.append({"lat": coords[0], "lon": coords[1], "Location": loc_name})
        st.map(pd.DataFrame(map_data), latitude='lat', longitude='lon')
        
        st.subheader("Recent Logs")
        st.dataframe(pd.DataFrame(data_log), use_container_width=True)

    elif st.session_state.nav_selection == "Add Sample Entry":
        st.markdown("<h3 style='text-align:center; margin-bottom: 15px;'>New Sample Entry</h3>", unsafe_allow_html=True)
        
        with st.form("sample_form"):
            st.subheader("Metadata")
            c1, c2 = st.columns(2)
            collection_date = c1.date_input("Date", datetime.date.today())
            collection_time = c2.time_input("Time", datetime.datetime.now().time())
            
            c3, c4 = st.columns(2)
            sampler_name = c3.text_input("Noted By:")
            location_select = c4.selectbox("Location", list(LOCATIONS.keys()))
            
            st.markdown("---")
            st.subheader("Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                ph = st.number_input("pH", 0.0, 20.0, 7.2)
                temp = st.number_input("Temp (°C)", 0.0, 50.0, 25.0)
                bod = st.number_input("BOD", 0.0, 100.0, 2.0)
                tss = st.number_input("TSS", 0.0, 1000.0, 30.0)
                fecal = st.number_input("Fecal Coliform", 0.0, 10000000.0, 100.0)

            with col2:
                do = st.number_input("DO", 0.0, 50.0, 6.0)
                chloride = st.number_input("Chloride", 0.0, 500.0, 50.0)
                phosphate = st.number_input("Phosphate", 0.0, 100.0, 0.1)
                color = st.number_input("Color", 0.0, 100.0, 15.0)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # UPDATED: Replaced use_container_width=True with width="stretch"
            submitted = st.form_submit_button("Save Record", width="stretch")
            
            if submitted:
                p_index = abs(ph - 7.0) + (tss / 10.0)
                input_df = pd.DataFrame([{
                    'ph': ph, 'Temperature': temp, 'BOD (mg/L)': bod, 'DO (mg/L)': do,
                    'Total Suspended Solids (mg/L)': tss, 'Chloride (mg/L)': chloride,
                    'Phosphate (mg/L)': phosphate, 'Color (TCU)': color,
                    'Fecal coliform (MPN/100mL)': fecal, 'Pollution_Index': p_index
                }])
                
                if model:
                    pred_idx = model.predict(input_df)[0]
                    proba = np.max(model.predict_proba(input_df))
                    try:
                        risk_label = encoder.inverse_transform([pred_idx])[0]
                    except:
                        risk_label = str(pred_idx)
                    
                    formatted_date = f"{collection_date} {collection_time.strftime('%I:%M %p')}"
                    
                    new_record = {
                        "Location": location_select,
                        "lat": LOCATIONS[location_select][0],
                        "lon": LOCATIONS[location_select][1],
                        "Risk": risk_label,
                        "Confidence": f"{proba:.0%}",
                        "Noted by:": sampler_name,
                        "Date": formatted_date,
                        "pH": ph,
                        "TSS": tss
                    }
                    save_new_entry(new_record)
                    
                    # --- PREDICTION MESSAGE UPDATE ---
                    if "High" in risk_label:
                        st.error(f"Result: {risk_label} ({proba:.0%}) - RECORD SAVED")
                    elif "Moderate" in risk_label:
                        st.warning(f"Result: {risk_label} ({proba:.0%}) - RECORD SAVED")
                    else:
                        st.success(f"Result: {risk_label} ({proba:.0%}) - RECORD SAVED")
                        st.info("ℹ️ Note: This water source is likely safe for consumption, but always remain cautious. Standard safety checks are still recommended.")
                else:
                    st.error("Model Error")

    # Logout
    st.markdown("<br><br>", unsafe_allow_html=True) 
    st.markdown("---")
    
    # UPDATED: Replaced use_container_width=True with width="stretch"
    if st.button("Logout", width="stretch"):
        logout()

# --- 6. USER VIEW ---
def user_view():
    data_log = load_data()

    st.markdown("<h1>Water Status</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Malaybalay City Risk Assessment</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Map Overview")
    
    map_data = []
    for loc_name, coords in LOCATIONS.items():
        map_data.append({"lat": coords[0], "lon": coords[1], "Location": loc_name})
    st.map(pd.DataFrame(map_data), latitude='lat', longitude='lon')
    
    st.markdown("---")

    st.subheader("Location Details")
    selected_loc = st.selectbox("Select Location", list(LOCATIONS.keys()))
    loc_history = [x for x in data_log if x['Location'] == selected_loc]
    
    if loc_history:
        latest = loc_history[-1] 
        risk = latest.get('Risk', 'Unknown')
        confidence = latest.get('Confidence', 'N/A')
        ph_val = latest.get('pH', 'N/A')
        tss_val = latest.get('TSS', 'N/A')
        date_val = latest.get('Date', 'N/A')
        
        color = get_risk_color(risk)
        
        # --- MESSAGE LOGIC FOR CARD ---
        recommendation = ""
        if 'High' in risk:
            recommendation = "CRITICAL: Water treatment required. Do not consume directly."
        elif 'Moderate' in risk:
            recommendation = "WARNING: Filtration or boiling recommended before use."
        else:
            # Low Risk Message
            recommendation = "SAFE: This water source is safe to consume, but always be cautious and check for physical changes."

        st.markdown(f"""
        <div style="
            padding: 20px; 
            border-radius: 20px; 
            border-left: 10px solid {color}; 
            background-color: white; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <h2 style="color: {color}; margin-top:0; text-align:center;">{risk}</h2>
            <p style="color: #666; font-size: 0.9rem; text-align:center;">Confidence: {confidence}</p>
            <div style="height: 1px; background-color: #eee; margin: 15px 0;"></div>
            <div style="display: flex; justify-content: space-between;">
                <div><p style="margin:0; font-weight:bold; color:#444;">pH</p><p style="margin:0; color:#666;">{ph_val}</p></div>
                <div><p style="margin:0; font-weight:bold; color:#444;">TSS</p><p style="margin:0; color:#666;">{tss_val}</p></div>
                <div><p style="margin:0; font-weight:bold; color:#444;">Date</p><p style="margin:0; color:#666;">{date_val.split(' ')[0]}</p></div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 10px; font-size: 0.9rem; text-align: center; color: #444;">
                <strong>Recommendation:</strong><br>{recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No data for {selected_loc}")

    # Logout
    st.markdown("<br><br>", unsafe_allow_html=True) 
    st.markdown("---")
    
    # UPDATED: Replaced use_container_width=True with width="stretch"
    if st.button("Logout", width="stretch"):
        logout()

# --- 7. MAIN APP ROUTER ---
if st.session_state.user_session == "admin":
    admin_view()
elif st.session_state.user_session == "user":
    user_view()
else:
    login()
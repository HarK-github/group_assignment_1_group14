import os
import sys
import time
import streamlit as st

# Add the current directory and parent to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from part1_live_editor import PassageSampler, MergeGenerator, EditorPipeline

# Page config
st.set_page_config(page_title="NLP Background Editor", layout="wide")

st.title("Integrated NLP Background Editor (Part 1)")
st.markdown("Simulated live-typing with real-time Segmentation, Spelling, and Grammar alerts.")

# Initialize Pipeline (cached to avoid reloading models on every UI interaction)
@st.cache_resource(show_spinner=True)
def load_pipeline():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    return EditorPipeline(models_dir)

pipeline = load_pipeline()

# Initialize Session State
if 'simulating' not in st.session_state:
    st.session_state.simulating = False
if 'sim_tokens' not in st.session_state:
    st.session_state.sim_tokens = []
if 'sim_idx' not in st.session_state:
    st.session_state.sim_idx = 0
if 'typed_text' not in st.session_state:
    st.session_state.typed_text = ""
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# --- Layout ---
col_main, col_sidebar = st.columns([2, 1])

with col_sidebar:
    st.header("Metrics & Controls")
    
    # Simulation Controls
    if st.button("Start Simulated Typing"):
        st.session_state.simulating = True
        st.session_state.sim_idx = 0
        st.session_state.typed_text = ""
        st.session_state.alerts = []
        pipeline.token_times = []
        pipeline.trigger_times = []
        pipeline.processed_words = []
        
        # Sample paragraph and generate tokens with simulated typing errors
        paragraph = PassageSampler.sample_paragraph()
        st.session_state.sim_tokens = MergeGenerator.simulate_typing(paragraph)
        st.rerun()
        
    if st.button("Stop Simulation"):
        st.session_state.simulating = False
        st.rerun()
        
    st.markdown("---")
    
    # Metrics
    st.subheader("Live Metrics")
    metric_placeholder = st.empty()
    
    def update_metrics():
        t_lat, trig_lat = pipeline.get_latencies()
        metric_placeholder.markdown(f"""
        - **Avg Token Check:** `{t_lat} ms`
        - **Avg Trigger Check:** `{trig_lat} ms`
        """)
        
    update_metrics()

with col_main:
    st.header("Live Typing Area")
    
    # Text Display
    text_placeholder = st.empty()
    text_placeholder.text_area("Document", value=st.session_state.typed_text, height=200, disabled=True)
    
    # Manual Input
    st.markdown("### Manual Live Input")
    st.caption("👉 **Manual Testing:** Type a sentence with missing spaces (e.g., 'helloworld') and press **Enter**.")
    def on_manual_input():
        new_text = st.session_state.manual_input
        if new_text.strip():
            words = new_text.strip().split()
            for word in words:
                result = pipeline.process_token(word)
                st.session_state.typed_text += result['token'] + " "
                if result['alert']:
                    st.session_state.alerts.insert(0, result['alert'])
            update_metrics()
        # Clear the input box after processing
        st.session_state.manual_input = ""
             
    st.text_input("Type words and press Enter to process:", key="manual_input", on_change=on_manual_input)
    
    # Alert Console
    st.header("Alert Console")
    alerts_placeholder = st.empty()
    
    def render_alerts():
        html = ""
        for alert in st.session_state.alerts:
            color = "#555" # Default gray
            if "[SEGMENT-ALERT]" in alert:
                color = "#d9534f" # Red
            elif "[SPELL-ALERT]" in alert:
                color = "#f0ad4e" # Orange
            elif "[GRAMMAR-ALERT]" in alert:
                color = "#5bc0de" # Light Blue
                
            html += f"<div style='padding: 8px; margin-bottom: 5px; border-left: 5px solid {color}; background-color: #f9f9f9; color: #333;'>{alert}</div>"
        
        if not html:
            html = "<div style='color: #888; font-style: italic;'>No alerts yet...</div>"
            
        alerts_placeholder.markdown(html, unsafe_allow_html=True)
        
    render_alerts()

# --- Simulation Loop ---
if st.session_state.simulating:
    if st.session_state.sim_idx < len(st.session_state.sim_tokens):
        # Get next token
        next_token = st.session_state.sim_tokens[st.session_state.sim_idx]
        st.session_state.sim_idx += 1
        
        # Process token through pipeline
        result = pipeline.process_token(next_token)
        
        # Update text (using the originally typed token, optionally highlight it if corrected, but we append corrected token space)
        # We'll display what the user "typed" literally, and alerts show corrections.
        st.session_state.typed_text += next_token + " "
        
        if result['alert']:
            # Add to top of list
            st.session_state.alerts.insert(0, result['alert'])
            
        # Update UI
        text_placeholder.text_area("Document", value=st.session_state.typed_text, height=200, disabled=True)
        update_metrics()
        render_alerts()
        
        # Sleep to simulate typing speed (e.g. 150-300ms per word)
        time.sleep(0.25)
        
        # Rerun to continue loop
        st.rerun()
    else:
        st.session_state.simulating = False
        st.success("Simulation Complete.")
        

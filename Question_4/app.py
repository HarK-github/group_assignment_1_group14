import os
import sys
import time
import streamlit as st

# Add the current directory and parent to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from part1_live_editor import PassageSampler, MergeGenerator, EditorPipeline

# Page config
st.set_page_config(page_title="NLP Background Editor", layout="wide")

st.title("Integrated NLP Background Editor")
st.markdown("Live-typing background editor hosting joint segmentation, spelling correction, smoothed N-gram language models, and PCFG constituency grammar analysis.")

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


# --- Part 3: Smoothed N-gram Sentence Analysis ---
st.markdown("---")
with st.expander("Part 3: Smoothed N-gram Sentence Scorer & Perplexity Inspector", expanded=False):
    st.markdown("Analyze sentences using Add-$k$ Smoothed Bigram and Trigram Language Models ($k=0.05$).")
    from part3_ngram import SmoothedNgramModels

    @st.cache_resource(show_spinner=False)
    def load_ngram_model():
        m_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        return SmoothedNgramModels.load_or_train(m_dir, k=0.05)

    ngram_model = load_ngram_model()

    inspect_text = st.text_area(
        "Sentence to inspect with smoothed N-gram models:",
        value=st.session_state.typed_text.strip() or "the quick brown fox jumps over the lazy dog",
        height=70,
        key="ngram_inspect_text"
    )
    if inspect_text.strip():
        tokens = [w.lower() for w in inspect_text.strip().split() if w.isalnum()]
        if tokens:
            bi_lp, bi_ppl = ngram_model.score_sentence_bigram(tokens)
            tri_lp, tri_ppl = ngram_model.score_sentence_trigram(tokens)

            col_bi, col_tri = st.columns(2)
            with col_bi:
                st.metric("Bigram Log-Prob", f"{bi_lp:.2f}")
                st.caption(f"Bigram Perplexity: `{bi_ppl:.2f}`")
            with col_tri:
                st.metric("Trigram Log-Prob", f"{tri_lp:.2f}")
                st.caption(f"Trigram Perplexity: `{tri_ppl:.2f}`")


# --- Part 4: Final Passage Analysis — Method Comparison Table ---
st.markdown("---")
st.header("Part 4: Final Passage Analysis — Method Comparison")
st.markdown(
    "Evaluates each sentence under three distinct models (**PCFG Constituency Parse**, "
    "**Add-$k$ Smoothed Bigram**, and **Add-$k$ Smoothed Trigram**) and applies the "
    "hierarchical decision rule to determine the final grammaticality verdict."
)

from part4_analysis import PassageAnalyzer

@st.cache_resource(show_spinner=False)
def load_passage_analyzer():
    return PassageAnalyzer()

analyzer = load_passage_analyzer()

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_analysis = st.button("Run Final Passage Analysis", type="primary", key="btn_run_part4")
with col_info:
    st.caption("Click to evaluate all sentences in the current document across PCFG and N-gram models.")

current_doc = st.session_state.typed_text.strip()
if run_analysis or (not st.session_state.simulating and len(st.session_state.alerts) > 0 and current_doc):
    with st.spinner("Analyzing passage structure and scoring sentences..."):
        input_data = pipeline.processed_words if pipeline.processed_words else current_doc
        if input_data:
            df_analysis = analyzer.analyze_passage(input_data)
            
            # Summary Metrics Row
            total_sents = len(df_analysis)
            n_grammatical = int((df_analysis['final verdict'] == 'Grammatical').sum())
            n_questionable = int((df_analysis['final verdict'] == 'Questionable').sum())
            n_ungrammatical = int((df_analysis['final verdict'] == 'Ungrammatical').sum())
            n_merges = int(df_analysis['number of segmentation merges resolved in this sentence'].sum())
            n_spells = int(df_analysis['number of spelling corrections applied in this sentence'].sum())
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Sentences", total_sents)
            m2.metric("Grammatical", n_grammatical)
            m3.metric("Questionable", n_questionable)
            m4.metric("Merges Resolved", n_merges)
            m5.metric("Spelling Corrections", n_spells)
            
            # Comparison Table Display
            st.subheader("Sentence-Level Comparison Summary Table")
            st.dataframe(df_analysis, use_container_width=True)
            
            # Decision Rule Expander
            with st.expander("Method Selection Decision Rule Logic", expanded=False):
                st.markdown("""
                - **Tier 1 (PCFG Parser):** Evaluates global phrase hierarchy (NP, VP, PP). If valid parse and normalized log-prob $\\ge -15.0$ nats/token $\\implies$ **PCFG (Grammatical)**.
                - **Tier 2 (Trigram LM):** Evaluates local 3-word fluency if PCFG is Unparseable or an outlier.
                  - If $\\ge -8.8$ nats/token $\\implies$ **Trigram LM (Grammatical)**.
                  - If between $[-11.0, -8.8)$ nats/token $\\implies$ **Trigram LM (Questionable)**.
                  - If $< -11.0$ nats/token $\\implies$ Defers to Tier 3.
                - **Tier 3 (Bigram LM):** Checks pairwise transitions. If $\\ge -8.5$ nats/token $\\implies$ **Bigram LM (Grammatical)**; else **Bigram LM (Ungrammatical)**.
                """)
        

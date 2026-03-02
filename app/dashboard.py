import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from wordcloud import WordCloud, STOPWORDS
import base64
import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# DARK MODE BACKGROUND FIX
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: #EAF3FB !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOR PALETTE — Green / Yellow / Red theme
# ─────────────────────────────────────────────
PALETTE = {
    'positive': '#5BAD7E',
    'neutral':  '#C49A3C',
    'negative': '#C0574E',
}
BG_PALE = {
    'positive': '#F2FAF5',
    'neutral':  '#FDF8EE',
    'negative': '#FDF2F1',
}
BORDER_COLOR = {
    'positive': '#5BAD7E',
    'neutral':  '#C49A3C',
    'negative': '#C0574E',
}
WORDCLOUD_COLORS = {
    'positive': '#5BAD7E',
    'neutral':  '#6B7280',
    'negative': '#C0574E'
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #EAF3FB;
    color: #1e293b;
}
header[data-testid="stHeader"] { display: none; }
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1200px;
    background-color: #EAF3FB;
}

/* Title */
.dash-title {
    font-size: 1.7rem; font-weight: 700;
    letter-spacing: -0.5px; color: #0f172a; 
    margin-bottom: 2rem;
    text-align: center;
}

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 120px;
    background: #f5f9fd;
    border: 1px solid #d0e4f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.total::before { background: #7b88d4; }
.kpi-card.pos::before   { background: #5BAD7E; }
.kpi-card.neg::before   { background: #C0574E; }
.kpi-card.neu::before   { background: #C49A3C; }

.kpi-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.9rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a; line-height: 1;
}
.kpi-pct { font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem; }
.kpi-icon { font-size: 1.3rem; float: right; opacity: 0.18; margin-top: -2rem; }

/* Section headings */
.section-head {
    font-size: 0.76rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #475569;
    margin: 1.3rem 0 0.6rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Summary box */
.summary-box {
    background: #f5f9fd;
    border: 1px solid #d0e4f0;
    border-left: 4px solid #7b88d4;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.87rem; line-height: 1.65;
    color: #334155; margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sentiment_results.csv"
SUMMARY_PATH = BASE_DIR / "data" / "summaries.json"

try:
    results = pd.read_csv(DATA_PATH)
    results['predicted_sentiment'] = results['predicted_sentiment'].str.lower()

    with open(SUMMARY_PATH) as f:
        summaries = json.load(f)
    emotion_summaries = summaries["emotion_summaries"]

    # ─────────────────────────────────────────────
    # SESSION STATE FIX
    # ─────────────────────────────────────────────
    if "summary_select" not in st.session_state:
        st.session_state.summary_select = list(emotion_summaries.keys())[0]
    if "wc_select" not in st.session_state:
        st.session_state.wc_select = "all"

    counts = results['predicted_sentiment'].value_counts()
    total = len(results)
    pos = counts.get('positive', 0)
    neg = counts.get('negative', 0)
    neu = counts.get('neutral', 0)
    sentiments = list(counts.index)

    # ─────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────
    st.markdown('<div class="dash-title">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # KPI CARDS
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card total">
        <div class="kpi-label">Total Comments</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-pct">All analysed records</div>
        <div class="kpi-icon">💬</div>
      </div>
      <div class="kpi-card pos">
        <div class="kpi-label">Positive Sentiment</div>
        <div class="kpi-value">{pos/total*100:.1f} %</div>
        <div class="kpi-pct">{pos:,} comments</div>
        <div class="kpi-icon">😊</div>
      </div>
      <div class="kpi-card neu">
        <div class="kpi-label">Neutral Sentiment</div>
        <div class="kpi-value">{neu/total*100:.1f} %</div>
        <div class="kpi-pct">{neu:,} comments</div>
        <div class="kpi-icon">😐</div>
      </div>
      <div class="kpi-card neg">
        <div class="kpi-label">Negative Sentiment</div>
        <div class="kpi-value">{neg/total*100:.1f} %</div>
        <div class="kpi-pct">{neg:,} comments</div>
        <div class="kpi-icon">😠</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        bar_colors_hex = [PALETTE.get(s, '#6366f1') for s in sentiments]
        pct_vals = [round(v / total * 100, 1) for v in counts.values]
        fig_bar = go.Figure(go.Bar(
            x=sentiments,
            y=list(counts.values),
            marker_color=bar_colors_hex,
            marker_line_width=0,
            text=[f'{p}%' for p in pct_vals],
            textposition='outside',
            textfont=dict(size=12, color='#334155', family='Sora, sans-serif'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata}%<extra></extra>',
            customdata=pct_vals,
        ))
        fig_bar.update_layout(
            title=dict(text='Sentiment Breakdown', font=dict(size=13, color='#334155', family='Sora, sans-serif'), x=0.5),
            paper_bgcolor='#f5f9fd',
            plot_bgcolor='#EAF3FB',
            font=dict(family='Sora, sans-serif', color='#334155'),
            xaxis=dict(showgrid=False, linecolor='#e2e8f0', tickfont=dict(size=12)),
            yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickfont=dict(size=10)),
            margin=dict(l=40, r=20, t=50, b=40),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
        fig2.patch.set_facecolor('#f5f9fd')
        ax2.set_facecolor('#f5f9fd')
        pie_colors = [PALETTE.get(s, '#6366f1') for s in sentiments]
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=None,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=2, edgecolor='white')
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color('white')
            at.set_fontweight('bold')
        legend_patches = [mpatches.Patch(color=PALETTE.get(s, '#6366f1'), label=s.capitalize()) for s in sentiments]
        ax2.legend(handles=legend_patches, loc='lower center', ncol=3,
                   framealpha=0, labelcolor='#334155', fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.1))
        ax2.set_title('Sentiment Distribution', color='#334155', fontsize=10, pad=6)
        fig2.tight_layout(pad=0.5)
        st.pyplot(fig2)
        plt.close(fig2)

    # ─────────────────────────────────────────────
    # OVERALL SUMMARY
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">🧾 Overall Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{summaries["overall_summary"]}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SENTIMENT-WISE SUMMARY
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">📝 Sentiment-wise Summary</div>', unsafe_allow_html=True)
    selected_sentiment = st.selectbox(
        "Select sentiment",
        options=list(emotion_summaries.keys()),
        label_visibility="collapsed",
        key="summary_select"
    )
    st.markdown(
        f'<div class="summary-box">{emotion_summaries[selected_sentiment]}</div>',
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────
    # WORD CLOUD (dynamic colors)
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Word Cloud</div>', unsafe_allow_html=True)
    wc_sentiment = st.selectbox(
        "Select sentiment for WordCloud",
        options=['all'] + [s for s in ['positive', 'neutral', 'negative'] if s in results['predicted_sentiment'].unique()],
        format_func=lambda x: x.capitalize(),
        label_visibility="collapsed",
        key="wc_select"
    )

    if "wc_image" not in st.session_state or st.session_state.get("wc_sentiment_cached") != wc_sentiment:
        if wc_sentiment == 'all':
            text_data = " ".join(results["comment_text"].dropna().astype(str))
            wc_bg_hex = '#f5f9fd'
            # Multi-color function
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return "hsl({}, 70%, 50%)".format(np.random.randint(0, 360))
        else:
            text_data = " ".join(results[results['predicted_sentiment'] == wc_sentiment]["comment_text"].dropna().astype(str))
            wc_bg_hex = BG_PALE.get(wc_sentiment, '#f5f9fd')
            color_hex = WORDCLOUD_COLORS.get(wc_sentiment, '#334155')
            # Single color function
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return color_hex

        if text_data.strip():
            wc = WordCloud(width=900, height=340, background_color=wc_bg_hex, stopwords=STOPWORDS, max_words=120, color_func=color_func).generate(text_data)
            fig_wc, ax_wc = plt.subplots(figsize=(9, 3.4))
            fig_wc.patch.set_facecolor(wc_bg_hex)
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.session_state["wc_image"] = fig_wc
            st.session_state["wc_sentiment_cached"] = wc_sentiment
            plt.close(fig_wc)
        else:
            st.warning("No comments available for this sentiment.")
            st.session_state["wc_image"] = None

    if st.session_state.get("wc_image"):
        st.pyplot(st.session_state["wc_image"])

    # ─────────────────────────────────────────────
    # DOWNLOAD CSV (centered green button)
    # ─────────────────────────────────────────────
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    b64_csv = base64.b64encode(csv_bytes).decode()
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1rem;">
        <a href="data:text/csv;base64,{b64_csv}" download="sentiment_analysis_results.csv"
           style="
                display: inline-block;
                padding: 0.55rem 1.2rem;
                font-size: 0.9rem;
                font-weight: 600;
                color: white;
                background-color: #16a34a;
                border-radius: 6px;
                text-decoration: none;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                transition: 0.2s;
           "
           onmouseover="this.style.backgroundColor='#15803d';"
           onmouseout="this.style.backgroundColor='#16a34a';"
        >
            ⬇ Download Complete Analysis Results
        </a>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Data files not found. Please ensure 'sentiment_results.csv' and 'summaries.json' are in the data folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")
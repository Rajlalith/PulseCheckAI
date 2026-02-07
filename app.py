"""
PulseCheck AI - Streamlit Dashboard Application

Interactive web dashboard for real-time sentiment analysis and customer
feedback intelligence using social media data.
"""

import os
import logging
import json
from datetime import datetime
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Import custom modules
from src.twitter_collector import TwitterCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import TopicModeler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PulseCheck AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for more app-like UI
st.markdown("""
    <style>
    /* Top navigation */
    .pc-header {
        display:flex;
        align-items:center;
        justify-content:space-between;
        background: linear-gradient(90deg, #0f6fc8 0%, #2aa198 100%);
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        box-shadow: 0 4px 14px rgba(15,111,200,0.12);
        margin-bottom: 16px;
    }
    .pc-title { font-size: 1.6rem; font-weight:700; }
    .pc-subtitle { font-size:0.9rem; opacity:0.9 }

    /* Metric cards */
    .pc-card { background:#ffffff; padding:14px; border-radius:10px; box-shadow: 0 6px 18px rgba(31,119,180,0.06); }
    .pc-metric { font-size:1.5rem; font-weight:700; color:#0f6fc8 }
    .pc-metric-label { color:#6b7280; margin-top:6px }

    /* Sidebar tweaks */
    .sidebar .stButton>button { width:100%; }

    /* Footer */
    .pc-footer { color:#94a3b8; font-size:0.85rem; margin-top:18px }
    </style>
""", unsafe_allow_html=True)

def render_top_header():
    left = "<div class='pc-title'>📊 PulseCheck AI</div>"
    right = f"<div class='pc-subtitle'>Real-Time social listening • {datetime.utcnow().strftime('%Y-%m-%d')}</div>"
    st.markdown(f"<div class='pc-header'>{left}{right}</div>", unsafe_allow_html=True)

def render_metric_card(label: str, value, help_text: str = ""):
    html = f"""
    <div class='pc-card'>
      <div class='pc-metric'>{value}</div>
      <div class='pc-metric-label'>{label}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


@st.cache_resource
def initialize_modules():
    """Initialize all processing modules."""
    twitter_collector = TwitterCollector(
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN")
    )
    sentiment_analyzer = SentimentAnalyzer()
    topic_modeler = TopicModeler()

    return twitter_collector, sentiment_analyzer, topic_modeler


def create_sentiment_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Create sentiment distribution pie chart."""
    if "sentiment" not in df.columns or len(df) == 0:
        return go.Figure()

    sentiment_counts = df["sentiment"].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(colors=["#d62728", "#2ca02c", "#7f7f7f"]),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        showlegend=True
    )

    return fig


def create_sentiment_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Create sentiment score distribution histogram."""
    if "sentiment_score" not in df.columns or len(df) == 0:
        return go.Figure()

    fig = go.Figure(data=[go.Histogram(
        x=df["sentiment_score"],
        nbinsx=30,
        marker_color="#1f77b4",
        hovertemplate="Score Range: %{x}<br>Count: %{y}<extra></extra>"
    )])

    fig.update_layout(
        title="Sentiment Score Distribution",
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        height=400
    )

    return fig


def create_emotion_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Create emotion distribution bar chart."""
    if "emotion" not in df.columns or len(df) == 0:
        return go.Figure()

    emotion_counts = df["emotion"].value_counts()

    fig = go.Figure(data=[go.Bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        marker_color="#1f77b4",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    )])

    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Emotion",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig


def create_emotion_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create emotion-sentiment cross-dimensional heatmap."""
    if "emotion" not in df.columns or "sentiment" not in df.columns:
        return go.Figure()

    heatmap_data = pd.crosstab(df["emotion"], df["sentiment"])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="Blues",
        hovertemplate="Emotion: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Emotion × Sentiment Heatmap",
        height=400
    )

    return fig


def create_topics_bar_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create top topics bar chart."""
    if "topics" not in df.columns or len(df) == 0:
        return go.Figure()

    topic_list = []
    for topics_str in df["topics"]:
        if isinstance(topics_str, str):
            topic_list.extend(topics_str.split(", "))

    topic_counts = pd.Series(topic_list).value_counts().head(top_n)

    fig = go.Figure(data=[go.Bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation="h",
        marker_color="#1f77b4",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    )])

    fig.update_layout(
        title="Top Keywords/Topics",
        xaxis_title="Frequency",
        height=400,
        showlegend=False
    )

    return fig


def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Create sentiment timeline chart."""
    if "created_at" not in df.columns or "sentiment" not in df.columns:
        return go.Figure()

    try:
        df_temp = df.copy()
        df_temp["created_at"] = pd.to_datetime(df_temp["created_at"])
        df_temp["date"] = df_temp["created_at"].dt.date

        sentiment_timeline = df_temp.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

        fig = go.Figure()

        for sentiment in sentiment_timeline.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_timeline.index,
                y=sentiment_timeline[sentiment],
                mode="lines+markers",
                name=sentiment.capitalize(),
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>"
            ))

        fig.update_layout(
            title="Sentiment Timeline",
            xaxis_title="Date",
            yaxis_title="Count",
            height=400,
            hovermode="x unified"
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        return go.Figure()


def export_data(df: pd.DataFrame, export_format: str):
    """Export analyzed data in specified format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if export_format == "CSV":
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"pulsecheck_analysis_{timestamp}.csv",
            mime="text/csv"
        )

    elif export_format == "JSON":
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"pulsecheck_analysis_{timestamp}.json",
            mime="application/json"
        )

    elif export_format == "Report":
        report = generate_text_report(df)
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"pulsecheck_report_{timestamp}.txt",
            mime="text/plain"
        )


def generate_text_report(df: pd.DataFrame) -> str:
    """Generate text summary report."""
    report = []
    report.append("=" * 60)
    report.append("PULSECHECK AI - ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("SUMMARY STATISTICS")
    report.append("-" * 60)
    report.append(f"Total Tweets Analyzed: {len(df)}")

    if "sentiment" in df.columns:
        sentiment_counts = df["sentiment"].value_counts()
        report.append(f"Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/len(df)*100:.1f}%)")
        report.append(f"Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/len(df)*100:.1f}%)")
        report.append(f"Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/len(df)*100:.1f}%)")

    if "emotion" in df.columns:
        report.append("")
        report.append("TOP EMOTIONS")
        report.append("-" * 60)
        emotion_counts = df["emotion"].value_counts().head(5)
        for emotion, count in emotion_counts.items():
            report.append(f"{emotion.capitalize()}: {count}")

    if "topics" in df.columns:
        report.append("")
        report.append("TOP KEYWORDS/TOPICS")
        report.append("-" * 60)
        topic_list = []
        for topics_str in df["topics"]:
            if isinstance(topics_str, str):
                topic_list.extend(topics_str.split(", "))

        topic_counts = pd.Series(topic_list).value_counts().head(10)
        for topic, count in topic_counts.items():
            report.append(f"{topic}: {count}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    """Main Streamlit application."""

    # Header (app-like)
    render_top_header()
    st.markdown("**Real-Time Social Media Sentiment Analysis & Customer Feedback Intelligence**")
    st.divider()

    # Initialize modules
    twitter_collector, sentiment_analyzer, topic_modeler = initialize_modules()

    # Sidebar configuration (compact form)
    with st.sidebar.form("config_form"):
        st.header("⚙️ Configuration")

        # Quick presets
        st.markdown("**Quick Queries**")
        preset = st.selectbox("Choose preset", ["#AI OR artificial intelligence", "#product", "@brandname", "custom"], index=0)

        # Data collection settings
        st.subheader("Data Collection")
        if preset != "custom":
            search_query = st.text_input("Search Query", value=preset)
        else:
            search_query = st.text_input("Search Query", value="#AI OR artificial intelligence")

        max_results = st.number_input(
            "Number of Tweets",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        days_back = st.selectbox("Days to Look Back", [1, 3, 5, 7], index=1)

        # Processing settings
        st.subheader("Processing")
        use_sample_data = st.checkbox("Use Sample Data (for testing without API)", value=True)

        collect_button = st.form_submit_button("🔍 Collect & Analyze Tweets")
        st.markdown("<div class='pc-footer'>Tip: Use sample data for quick demos.</div>", unsafe_allow_html=True)

    # Main content area
    if collect_button:
        with st.spinner("Collecting tweets..."):
            if use_sample_data:
                df = twitter_collector.search_sample_data(search_query, n_samples=max_results)
                st.info("ℹ️ Using sample data for demonstration")
            else:
                df = twitter_collector.collect_tweets(search_query, max_results, days_back)

        if df is not None and len(df) > 0:
            with st.spinner("Analyzing sentiment and emotions..."):
                df = sentiment_analyzer.analyze_batch(df)

            with st.spinner("Extracting topics..."):
                df = topic_modeler.analyze_topics(df)

            # Store in session state
            st.session_state.analyzed_df = df
            st.success(f"✅ Analysis complete! Processed {len(df)} tweets")
        else:
            st.error("❌ No tweets found. Try different search parameters.")

    # Display analysis if data exists
    if "analyzed_df" in st.session_state:
        df = st.session_state.analyzed_df

        # Tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📈 Sentiment Overview", "😊 Emotion Analysis", "🏷️ Topic Modeling", "📊 Trends", "💾 Export"]
        )

        # Tab 1: Sentiment Overview
        with tab1:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                render_metric_card("Total Tweets", len(df))

            if "sentiment" in df.columns:
                sentiment_summary = sentiment_analyzer.get_sentiment_summary(df)

                with col2:
                    render_metric_card("Positive", sentiment_summary["positive_count"], "% positive")

                with col3:
                    render_metric_card("Negative", sentiment_summary["negative_count"], "% negative")

                with col4:
                    render_metric_card("Neutral", sentiment_summary["neutral_count"], "% neutral")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(create_sentiment_pie_chart(df), use_container_width=True)

            with col2:
                st.plotly_chart(create_sentiment_score_histogram(df), use_container_width=True)

        # Tab 2: Emotion Analysis
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(create_emotion_bar_chart(df), use_container_width=True)

            with col2:
                st.plotly_chart(create_emotion_sentiment_heatmap(df), use_container_width=True)

        # Tab 3: Topic Modeling
        with tab3:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.plotly_chart(create_topics_bar_chart(df), use_container_width=True)

            with col2:
                st.subheader("Topic Summary")
                topics_summary = topic_modeler.get_topics_summary(df)
                for topic, count in list(topics_summary.items())[:10]:
                    st.write(f"**{topic}**: {count}")

        # Tab 4: Trends Over Time
        with tab4:
            st.plotly_chart(create_timeline_chart(df), use_container_width=True)

        # Tab 5: Export
        with tab5:
            st.subheader("📥 Export Data")

            export_format = st.radio(
                "Select export format:",
                ["CSV", "JSON", "Report"],
                horizontal=True
            )

            export_data(df, export_format)

            st.divider()

            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()

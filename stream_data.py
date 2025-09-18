# app.py
import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import pycountry

DB_PATH = os.getenv("DB_PATH", "countries.db")

# ---------- Setup ----------
st.set_page_config(page_title="Country Survey", page_icon="🌍", layout="wide")

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            country_name TEXT NOT NULL,
            alpha3 TEXT NOT NULL
        )
    """)
    return conn

conn = get_conn()

@st.cache_data(ttl=5)
def load_data():
    return pd.read_sql_query("SELECT * FROM responses", conn)

# country lists & helpers
@st.cache_data
def get_countries():
    # Use official country names from pycountry (no historical/obsolete)
    rows = []
    for c in pycountry.countries:
        rows.append({"name": c.name, "alpha_3": getattr(c, "alpha_3", None)})
    df = pd.DataFrame(rows).dropna(subset=["alpha_3"]).drop_duplicates("alpha_3")
    df = df.sort_values("name", key=lambda s: s.str.normalize('NFKD'))
    return df

countries_df = get_countries()
country_names = countries_df["name"].tolist()
alpha3_map = dict(zip(countries_df["name"], countries_df["alpha_3"]))

# ---------- UI Navigation ----------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Survey", "📄 Data", "📊 Visualizations", "🧠 Insights"])

# ---------- Survey Tab ----------
with tab1:
    st.header("Where are you from?")
    st.write("Choose your country and submit. Results update live.")
    with st.form("survey_form", clear_on_submit=True):
        chosen = st.selectbox("Country", country_names, index=country_names.index("United States") if "United States" in country_names else 0)
        submitted = st.form_submit_button("Submit response")
    if submitted and chosen:
        conn.execute(
            "INSERT INTO responses (ts, country_name, alpha3) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), chosen, alpha3_map[chosen])
        )
        conn.commit()
        st.success(f"Thanks! Recorded: {chosen}")

# ---------- Data Tab ----------
with tab2:
    st.header("Raw Counts")
    df = load_data()

    if df.empty:
        st.info("No responses yet. Ask participants to submit in the **Survey** tab.")
    else:
        counts = (
            df.groupby(["country_name", "alpha3"], as_index=False)
              .size()
              .rename(columns={"size": "count"})
              .sort_values("count", ascending=False)
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Total responses", int(df.shape[0]))
            st.metric("Unique countries", int(counts.shape[0]))
            top = counts.head(1)
            if not top.empty:
                st.metric("Most common country", f"{top.iloc[0]['country_name']} ({top.iloc[0]['count']})")
        with c2:
            st.dataframe(counts, use_container_width=True)

# ---------- Visualizations Tab ----------
with tab3:
    st.header("Charts & Map")
    df = load_data()
    if df.empty:
        st.info("No data to visualize yet.")
    else:
        counts = (
            df.groupby(["country_name", "alpha3"], as_index=False)
              .size()
              .rename(columns={"size": "count"})
              .sort_values("count", ascending=False)
        )

        st.subheader("Bar chart (Top 20)")
        topn = st.slider("How many countries to show", 5, min(50, len(counts)), min(1, len(counts)))
        bar = px.bar(
            counts.head(topn),
            x="country_name",
            y="count",
            labels={"country_name": "Country", "count": "Responses"},
            title=None
        )
        bar.update_layout(xaxis_tickangle=-35, margin=dict(l=10, r=10, t=10, b=10), height=450)
        st.plotly_chart(bar, use_container_width=True)

        st.subheader("Word cloud")
        freq = {row["country_name"]: int(row["count"]) for _, row in counts.iterrows()}
        wc = WordCloud(width=1200, height=600, background_color="white")
        wc_img = wc.generate_from_frequencies(freq)
        buf = io.BytesIO()
        wc_img.to_image().save(buf, format="PNG")
        st.image(buf.getvalue(), caption="Country frequency word cloud")

        st.subheader("World map")
        # plotly choropleth expects ISO-3 in 'locations'
        map_fig = px.choropleth(
            counts,
            locations="alpha3",
            color="count",
            hover_name="country_name",
            color_continuous_scale="Viridis",
            projection="natural earth",
        )
        map_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520)
        st.plotly_chart(map_fig, use_container_width=True)

# ---------- Insights Tab ----------
with tab4:
    st.header("AI-style Insights (local heuristic)")
    df = load_data()
    if df.empty:
        st.info("No data yet. Insights will appear once there are responses.")
    else:
        counts = (
            df.groupby(["country_name"], as_index=False)
              .size()
              .rename(columns={"size": "count"})
              .sort_values("count", ascending=False)
        )
        total = int(df.shape[0])
        unique = int(counts.shape[0])
        top3 = counts.head(3).to_dict(orient="records")

        # Diversity proxy (normalized inverse Herfindahl index)
        p = counts["count"] / total
        hhi = float((p ** 2).sum())
        diversity = (1 / hhi) if hhi > 0 else 0

        lines = []
        lines.append(f"- **Total responses:** {total} across **{unique}** countries.")
        if len(top3) >= 1:
            top_str = ", ".join([f"{r['country_name']} ({r['count']})" for r in top3])
            lines.append(f"- **Top countries:** {top_str}.")
        if diversity >= 8:
            lines.append("- **High diversity**: responses are broadly distributed across many countries.")
        elif diversity >= 4:
            lines.append("- **Moderate diversity**: a healthy mix with a few standouts.")
        else:
            lines.append("- **Low diversity**: responses are concentrated in a small set of countries.")

        # Trend note (last 10 vs. previous 10, if available)
        if len(df) >= 20:
            last10 = df.tail(10).country_name.value_counts()
            prev10 = df.tail(20).head(10).country_name.value_counts()
            gaining = (last10 - prev10).sort_values(ascending=False)
            gaining = gaining[gaining > 0]
            if not gaining.empty:
                lines.append(f"- **Recent uptick** from: {', '.join(gaining.index[:3])}.")

        st.markdown("##### Summary")
        st.markdown("\n".join(lines))

        st.markdown("##### Suggestions")
        st.markdown(
            "- Encourage underrepresented regions to improve balance.\n"
            "- Keep the survey open longer to increase sample size.\n"
            "- Consider segmenting results by event/session if applicable."
        )

        st.caption("Note: This is a lightweight, local heuristic summary (no external API).")


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
import os

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Survey", "📄 Data", "📊 Visualizations", "🧠 Insights", "🤖 AI Copy (≤250 chars)"])

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

        # --- Bar chart (Top N) ---
        st.subheader("Bar chart (Top N)")
        n = len(counts)
        max_n = max(1, min(50, n))
        default_n = min(20, max_n)
        if n <= 1:
            topn = n
        else:
            topn = st.slider(
                "How many countries to show",
                min_value=1, max_value=max_n, value=default_n, step=1
            )

        bar_fig = px.bar(
            counts.head(topn),
            x="country_name",
            y="count",
            labels={"country_name": "Country", "count": "Responses"},
            title=None
        )
        bar_fig.update_layout(xaxis_tickangle=-35, margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(bar_fig, use_container_width=True)

        # --- Word cloud + quick table ---
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Word cloud")
            freq = {row["country_name"]: int(row["count"]) for _, row in counts.iterrows()}
            if sum(freq.values()) == 0:
                st.info("Not enough data for a word cloud yet.")
            else:
                from wordcloud import WordCloud
                import io
                wc = WordCloud(width=1200, height=600, background_color="white")
                wc_img = wc.generate_from_frequencies(freq)
                buf = io.BytesIO()
                wc_img.to_image().save(buf, format="PNG")
                st.image(buf.getvalue(), caption="Country frequency word cloud", use_column_width=True)

        with right:
            st.subheader("Top countries")
            st.dataframe(
                counts[["country_name", "count"]].head(10).rename(columns={"country_name": "Country", "count": "Responses"}),
                use_container_width=True,
                height=320
            )

        # --- World map ---
        st.subheader("World map")
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

# ---------- Insights Tab (statistical) ----------
with tab4:
    st.header("Statistical Insights (no external AI)")
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
        st.write(f"Sample size: **n = {total}** | Unique countries: **{unique}**")

        # Helpers
        import math

        def wilson_ci(k, n, z=1.96):
            if n == 0:
                return (0.0, 0.0)
            p = k / n
            denom = 1 + z**2 / n
            center = (p + z**2/(2*n)) / denom
            half_width = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            return (max(0.0, center - half_width), min(1.0, center + half_width))

        def two_prop_z(k1, n1, k2, n2):
            # Pooled two-proportion z-test for p1 - p2
            if min(n1, n2) == 0:
                return None, None
            p1, p2 = k1/n1, k2/n2
            p_pool = (k1 + k2) / (n1 + n2)
            se = math.sqrt(p_pool*(1-p_pool) * (1/n1 + 1/n2))
            if se == 0:
                return None, None
            z = (p1 - p2) / se
            # two-sided p-value from normal approximation
            from math import erf, sqrt
            def norm_cdf(x):  # N(0,1)
                return 0.5 * (1 + erf(x / sqrt(2)))
            pval = 2 * (1 - norm_cdf(abs(z)))
            return z, pval

        def one_prop_z_greater(k, n, p0=0.5):
            # H0: p = p0 vs H1: p > p0 (normal approx)
            if n == 0:
                return None, None
            p_hat = k/n
            se = math.sqrt(p0*(1-p0)/n)
            if se == 0:
                return None, None
            z = (p_hat - p0)/se
            # one-sided p-value
            from math import erf, sqrt
            def norm_cdf(x):
                return 0.5 * (1 + erf(x / sqrt(2)))
            pval = 1 - norm_cdf(z)
            return z, pval

        # Diversity proxy: inverse Herfindahl (higher => more diverse)
        p = counts["count"] / total
        hhi = float((p**2).sum())
        diversity = (1/hhi) if hhi > 0 else 0

        lines = []
        caveats = []

        # Top country insight
        top_row = counts.iloc[0]
        top_country = top_row["country_name"]
        top_k = int(top_row["count"])
        top_share = top_k / total
        lo, hi = wilson_ci(top_k, total)

        lines.append(
            f"- **Most represented:** {top_country} with **{top_k}/{total} ({top_share:.1%})** "
            f"[95% CI: {lo:.1%}–{hi:.1%}]."
        )

        # Top vs 50% majority test (only meaningful if n >= 30)
        if total >= 30:
            z, pval = one_prop_z_greater(top_k, total, p0=0.5)
            if z is not None:
                if pval < 0.05:
                    lines.append(f"- **Majority test:** proportion from {top_country} is **> 50%** (z = {z:.2f}, p = {pval:.3f}).")
                else:
                    lines.append(f"- **Majority test:** no evidence that {top_country} exceeds **50%** (z = {z:.2f}, p = {pval:.3f}).")
        else:
            caveats.append("Majority test skipped (needs n ≥ 30 for a reliable normal approximation).")

        # Top vs runner-up comparison
        if len(counts) >= 2:
            second = counts.iloc[1]
            c2, k2 = second["country_name"], int(second["count"])
            z2, p2 = two_prop_z(top_k, total, k2, total)
            if z2 is not None:
                if p2 < 0.05:
                    lines.append(f"- **Lead over runner-up:** {top_country} > {c2} (z = {z2:.2f}, p = {p2:.3f}).")
                else:
                    lines.append(f"- **Lead over runner-up:** difference is **not significant** (z = {z2:.2f}, p = {p2:.3f}).")

        # Diversity interpretation
        if diversity >= 8:
            lines.append("- **Diversity:** High (responses spread across many countries).")
        elif diversity >= 4:
            lines.append("- **Diversity:** Moderate (mix of countries with a few standouts).")
        else:
            lines.append("- **Diversity:** Low (responses concentrated in a few countries).")

        # Recent trend (last 10 vs previous 10), if possible
        if len(df) >= 20:
            last10 = df.tail(10).country_name.value_counts()
            prev10 = df.tail(20).head(10).country_name.value_counts()
            gaining = (last10 - prev10).sort_values(ascending=False)
            gaining = gaining[gaining > 0]
            if not gaining.empty:
                lines.append(f"- **Recent uptick:** {', '.join(gaining.index[:3])}.")

        st.markdown("##### Insights")
        st.markdown("\n".join(lines))

        if caveats:
            st.markdown("##### Notes & Caveats")
            st.markdown("\n".join(f"- {c}" for c in caveats))

        st.caption(
            "Stats use Wilson CIs and large-sample z-tests (α = 0.05). "
            "Interpret as insights about *this sample*, not the entire population."
        )

# ---------- AI Copy Tab (≤250 chars) ----------


# ---------- AI Copy Tab (≤250 chars) ----------
# ---------- AI Audience Summary Tab (1–2 paragraphs) ----------
with tab5:
    st.header("AI Audience Summary")

    df = load_data()
    if df.empty:
        st.info("No data yet. Submit responses first.")
    else:
        # Prepare country counts
        counts = (
            df.groupby(["country_name"], as_index=False)
              .size()
              .rename(columns={"size": "count"})
              .sort_values("count", ascending=False)
        )
        total = int(df.shape[0])
        unique = int(counts.shape[0])
        top_list = counts.head(5).to_dict(orient="records")

        # Simple diversity proxy (inverse Herfindahl)
        import math
        p = (counts["count"] / total) if total else pd.Series(dtype=float)
        hhi = float((p**2).sum()) if total else 0.0
        diversity = (1 / hhi) if hhi > 0 else 0.0
        if diversity >= 8:
            diversity_tag = "high diversity"
        elif diversity >= 4:
            diversity_tag = "moderate diversity"
        else:
            diversity_tag = "low diversity"

        # Recent uptick (last 10 vs. previous 10)
        uptick = ""
        if len(df) >= 20:
            last10 = df.tail(10).country_name.value_counts()
            prev10 = df.tail(20).head(10).country_name.value_counts()
            gaining = (last10 - prev10).sort_values(ascending=False)
            gaining = [c for c, v in gaining.items() if v > 0]
            if gaining:
                uptick = ", ".join(gaining[:3])

        # Controls
        col1, col2 = st.columns([1,1])
        with col1:
            tone = st.selectbox("Tone", ["Professional", "Friendly", "Energetic", "Neutral"], index=0)
        with col2:
            paragraphs = st.radio("Length", ["1 paragraph", "2 paragraphs"], index=1)

        # Build factual context for the model
        breakdown = ", ".join(
            f"{r['country_name']} ({int(r['count'])}, {int(round(r['count']*100/total))}%)"
            for r in top_list
        ) if total else "—"

        context = {
            "total_responses": total,
            "unique_countries": unique,
            "top_breakdown": breakdown,
            "diversity": diversity_tag,
            "recent_uptick": uptick or None,
        }

        def local_summary():
            # Local fallback: 1–2 short paragraphs, no external API
            top_name = counts.iloc[0]["country_name"] if not counts.empty else "—"
            top_share = (counts.iloc[0]["count"] / total) if total else 0
            p1 = (
                f"We’ve gathered {total} responses across {unique} countries. "
                f"{top_name} currently leads with {top_share:.0%} of the sample. "
                f"Top countries so far: {breakdown}. Overall reach shows {diversity_tag}."
            )
            p2 = ""
            if paragraphs == "2 paragraphs":
                if context["recent_uptick"]:
                    p2 = f"Recent momentum appears from {context['recent_uptick']}. We’ll keep tracking as new responses arrive."
                else:
                    p2 = "We’ll keep tracking how the mix evolves as more participants join."
            return p1 if not p2 else p1 + "\n\n" + p2

        # Generate with OpenAI when available
        generate = st.button("Generate summary")
        summary = ""

        if generate:
            use_openai = bool(os.getenv("OPENAI_API_KEY"))
            if use_openai:
                try:
                    from openai import OpenAI
                    client = OpenAI()

                    length_req = "one paragraph" if paragraphs == "1 paragraph" else "two short paragraphs"
                    prompt = (
                        "Write {length_req} (≈80–160 words total) summarizing audience origins for a LinkedIn update. "
                        "Be factual, concise, and positive. No emojis, no hashtags, no URLs. "
                        "Mention total responses, top countries, and overall diversity; optionally note any recent uptick.\n\n"
                        "Facts (use as ground truth):\n"
                        f"- Total responses: {context['total_responses']}\n"
                        f"- Unique countries: {context['unique_countries']}\n"
                        f"- Top countries: {context['top_breakdown']}\n"
                        f"- Diversity: {context['diversity']}\n"
                        f"- Recent uptick: {context['recent_uptick']}\n"
                        f"- Tone: {tone}\n"
                    ).format(length_req=length_req)

                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=400,
                    )
                    summary = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.warning(f"OpenAI generation failed: {e}. Showing local summary instead.")
                    summary = local_summary()
            else:
                st.info("OPENAI_API_KEY not set — using local summary.")
                summary = local_summary()

        # Show result
        if summary:
            st.subheader("Audience summary")
            st.markdown(summary)
            st.download_button(
                "Download summary (.txt)",
                data=summary.encode("utf-8"),
                file_name="audience_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )





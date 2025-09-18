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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Survey", "📄 Data", "📊 Visualizations", "🧠 Insights", "AI post"])

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
# --- put near your other imports ---
import os

# ---------- AI Copy Tab (≤250 chars) ----------
tab5, = st.tabs(["🤖 AI Copy (≤250 chars)"])

with tab5:
    st.header("AI-style social snippets (≤250 chars)")
    df = load_data()

    if df.empty:
        st.info("No data yet. Submit responses first.")
    else:
        counts = (
            df.groupby(["country_name"], as_index=False)
              .size()
              .rename(columns={"size": "count"})
              .sort_values("count", ascending=False)
        )
        total = int(df.shape[0])
        top = counts.iloc[0] if not counts.empty else None
        top_name = top["country_name"] if top is not None else "—"
        top_share = (top["count"] / total) if top is not None else 0.0

        # UI options
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            add_emojis = st.checkbox("Add emojis", value=True)
        with c2:
            add_hashtags = st.checkbox("Add hashtags", value=False)
        with c3:
            n_variants = st.slider("Variants", 1, 5, 3)

        # Context string for prompting or local fallback
        context = (
            f"Total responses: {total}. "
            f"Top country: {top_name} ({top_share:.1%}). "
            "Other notable: " +
            ", ".join([f"{r.country_name} ({int(r['count'])})" for _, r in counts.head(5).iterrows()])
        )

        # ---------- Local heuristic generator (no external API) ----------
        def local_variants(k):
            base_bits = []
            if add_emojis:
                base_bits.append("🌍")
            prefix = " ".join(base_bits)
            msg1 = f"{prefix} From our sample of {total}, {top_name} leads at {top_share:.0%}."
            msg2 = f"{prefix} Most respondents are from {top_name}; results keep rolling in."
            msg3 = f"{prefix} Snapshot: {top_name} tops the list; more countries represented soon."
            pool = [msg1, msg2, msg3]

            # Expand if more needed
            while len(pool) < k:
                i = len(pool) + 1
                pool.append(f"{prefix} Update #{i}: {top_name} still ahead; {total} responses so far.")

            def decorate(s: str) -> str:
                tags = " #survey #community" if add_hashtags else ""
                out = (s + tags).strip()
                return (out[:247] + "…") if len(out) > 250 else out

            return [decorate(x) for x in pool[:k]]

        # ---------- Optional OpenAI generation ----------
        use_openai = st.toggle("Use OpenAI (set OPENAI_API_KEY)", value=False)
        ai_texts = []

        if use_openai and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()  # uses OPENAI_API_KEY from env
                style = "energetic, clear, suitable for social media, no URLs."
                extras = []
                if add_emojis:
                    extras.append("tasteful emojis allowed")
                if add_hashtags:
                    extras.append("1–2 short hashtags allowed")
                extras = ", ".join(extras) if extras else "no emojis or hashtags"

                prompt = (
                    "You generate SHORT social updates (<=250 characters). "
                    "Write tight, positive, factual lines about survey results.\n"
                    f"Context: {context}\n"
                    f"Tone: {style}\n"
                    f"Extras: {extras}\n"
                    f"Return {n_variants} distinct lines, each on its own line, no numbering."
                )

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=220,
                )
                raw = resp.choices[0].message.content.strip().split("\n")
                for line in raw:
                    txt = line.strip("•- \t")
                    if not add_hashtags:
                        parts = [w for w in txt.split() if not (w.startswith("#") and len(w) > 15)]
                        txt = " ".join(parts)
                    txt = (txt[:247] + "…") if len(txt) > 250 else txt
                    if txt:
                        ai_texts.append(txt)

                # Ensure exactly n_variants
                if len(ai_texts) < n_variants:
                    ai_texts += local_variants(n_variants - len(ai_texts))
                else:
                    ai_texts = ai_texts[:n_variants]

            except Exception as e:
                st.warning(f"OpenAI generation failed: {e}. Falling back to local.")
                ai_texts = local_variants(n_variants)
        else:
            ai_texts = local_variants(n_variants)

        st.subheader("Your snippets")
        for i, t in enumerate(ai_texts, 1):
            st.code(t, language=None)  # handy copy button

        # Download all as a text file
        all_txt = "\n\n".join(ai_texts)
        st.download_button(
            "Download snippets (.txt)",
            data=all_txt.encode("utf-8"),
            file_name="ai_snippets.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption("Tip: toggle OpenAI for punchier copy. Without a key, the app uses a local generator.")



# import ollama
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime

# Optional AI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# PDF reports
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="InsightForgeAI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at top right, #173B75 0%, transparent 35%),
        linear-gradient(135deg, #07111F 0%, #0B1D35 50%, #102A56 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: #07111F;
    border-right: 1px solid #1E3A5F;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3, h4, h5, p, label {
    color: white !important;
}

.sidebar-brand {
    text-align: center;
    padding: 8px 0 16px 0;
}

.sidebar-brand img {
    border-radius: 14px;
    margin-bottom: 8px;
}

.sidebar-brand-name {
    color: #60A5FA;
    font-size: 21px;
    font-weight: 800;
}

.workspace-title {
    color: #93C5FD !important;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 8px 0;
}

/* Left workspace buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 42px;
    justify-content: flex-start;
    text-align: left;
    border: 1px solid transparent;
    border-radius: 9px;
    background: transparent;
    color: #E2E8F0 !important;
    font-weight: 600;
    padding: 8px 12px;
    margin: 2px 0;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #122B4A;
    border-color: #214D7A;
    color: white !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #173B75;
    border-color: #2B6CB0;
    color: white !important;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #2563EB;
    background: #123B70;
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1D4ED8;
}

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #7DD3FC !important;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}

.powered {
    text-align: center;
    color: #CBD5E1 !important;
    margin-bottom: 28px;
}

.hero-logo {
    display: block;
    margin: 8px auto 12px auto;
    border-radius: 20px;
}

.card {
    background: rgba(15, 35, 65, 0.88);
    border: 1px solid #244B78;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(135deg, #102A43, #163D68);
    border: 1px solid #285B8F;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #60A5FA;
}

.metric-label {
    color: #CBD5E1;
    font-size: 14px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #93C5FD;
    margin-top: 20px;
}

.small-muted {
    color: #94A3B8 !important;
}

.footer {
    text-align: center;
    color: #94A3B8 !important;
    margin-top: 60px;
    padding: 20px 0;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 35, 65, 0.7);
    border: 1px solid #244B78;
    border-radius: 14px;
    padding: 10px;
}

div[data-testid="stMetric"] {
    background: rgba(15, 35, 65, 0.75);
    border: 1px solid #244B78;
    padding: 12px;
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "df": None,
    "file_name": None,
    "messages": [],
    "history": [],
    "page": "New Analysis",
    "analysis_name": "New Analysis",
    "api_key": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def load_dataset(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")


def clean_dataframe(df):
    df = df.copy()
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_dataset_context(df):
    numeric = df.select_dtypes(include=np.number)

    context = [
        f"Dataset rows: {len(df)}",
        f"Dataset columns: {len(df.columns)}",
        f"Column names: {list(df.columns)}",
        "\nData types:",
        df.dtypes.astype(str).to_string(),
        "\nMissing values:",
        df.isnull().sum().to_string(),
    ]

    if len(numeric.columns) > 0:
        context.extend(
            [
                "\nNumeric statistics:",
                numeric.describe().round(3).to_string(),
            ]
        )

    # Keep the prompt reasonably sized.
    preview = df.head(100)
    context.extend(
        [
            "\nFirst 100 rows:",
            preview.to_csv(index=False),
        ]
    )

    return "\n".join(context)


def local_answer(question, df):
    """Useful local dataset answers when no AI key is configured."""

    q = question.lower().strip()

    if ("row" in q or "record" in q) and ("how many" in q or "number" in q):
        return f"The dataset contains **{len(df):,} rows**."

    if "column" in q and ("how many" in q or "number" in q):
        return f"The dataset contains **{len(df.columns)} columns**."

    if "missing" in q or "null" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            return "There are **no missing values** in the dataset."

        return "Missing values found:\n\n" + missing.to_string()

    if "column" in q or "columns" in q:
        return "The available columns are:\n\n" + "\n".join(
            f"- `{c}`" for c in df.columns
        )

    if "summary" in q or "summarize" in q:
        return (
            "### Dataset Summary\n\n"
            f"- Rows: **{len(df):,}**\n"
            f"- Columns: **{len(df.columns)}**\n"
            f"- Numeric columns: **{len(get_numeric_columns(df))}**\n"
            f"- Missing values: **{int(df.isnull().sum().sum()):,}**"
        )

    numeric = get_numeric_columns(df)

    for column in numeric:
        if str(column).lower() in q:
            if "average" in q or "mean" in q:
                return f"The average of **{column}** is **{df[column].mean():,.2f}**."

            if "maximum" in q or "highest" in q or "max" in q:
                return f"The maximum of **{column}** is **{df[column].max():,.2f}**."

            if "minimum" in q or "lowest" in q or "min" in q:
                return f"The minimum of **{column}** is **{df[column].min():,.2f}**."

    return (
        "I can answer basic questions locally. For full natural-language "
        "AI analysis, add an OpenAI API key in Settings."
    )


def ask_ai(question, df):
    """AI-powered business data analysis using Google Gemini."""

    try:
        from google import genai

        rows = len(df)
        columns = list(df.columns)

        data_preview = df.head(50).to_string(index=False)

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:
            numeric_summary = (
                df[numeric_cols]
                .describe()
                .round(2)
                .to_string()
            )
        else:
            numeric_summary = "No numerical columns available."

        prompt = f"""
You are InsightForgeAI, a professional business data analysis assistant.

Answer the user's question using ONLY the uploaded dataset information.

RULES:
- Understand natural-language questions.
- Use only the dataset information provided.
- Do not invent numbers.
- Calculate results from the dataset when possible.
- If the dataset does not contain the required information, clearly say so.
- Give clear and professional answers.
- Provide useful business insights when appropriate.

DATASET INFORMATION

Rows:
{rows}

Columns:
{columns}

DATA PREVIEW:
{data_preview}

NUMERICAL SUMMARY:
{numeric_summary}

USER QUESTION:
{question}

Answer the user's question using the dataset above.
"""

        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            return "Gemini API key is not configured."

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Gemini AI error: {str(e)}"


def generate_insights(df):
    insights = []

    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isnull().sum().sum())

    insights.append(
        f"The dataset contains **{rows:,} rows** and **{cols} columns**."
    )

    if missing == 0:
        insights.append("There are **no missing values** in the dataset.")
    else:
        insights.append(
            f"The dataset contains **{missing:,} missing values**."
        )

    numeric = df.select_dtypes(include=np.number)

    for column in numeric.columns:
        try:
            mean = numeric[column].mean()
            maximum = numeric[column].max()
            minimum = numeric[column].min()

            insights.append(
                f"**{column}** ranges from **{minimum:,.2f}** to "
                f"**{maximum:,.2f}**, with an average of **{mean:,.2f}**."
            )
        except Exception:
            pass

    return insights


def create_pdf(df):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = [
        Paragraph("InsightForgeAI — Data Analysis Report", title_style),
        Spacer(1, 20),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            styles["Normal"],
        ),
        Spacer(1, 15),
        Paragraph(
            f"Rows: {len(df):,} &nbsp;&nbsp;&nbsp; Columns: {len(df.columns)}",
            styles["Normal"],
        ),
        Spacer(1, 15),
    ]

    for insight in generate_insights(df):
        story.append(Paragraph(insight.replace("**", ""), styles["Normal"]))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 15))
    story.append(Paragraph("Column Information", styles["Heading2"]))

    table_data = [["Column", "Data Type", "Missing"]]

    for column in df.columns:
        table_data.append(
            [
                str(column),
                str(df[column].dtype),
                str(int(df[column].isnull().sum())),
            ]
        )

    table = Table(table_data)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B70")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(table)
    doc.build(story)

    buffer.seek(0)
    return buffer


def show_front_page():
    """Professional landing/new-analysis page."""

    logo_path = os.path.join("assets", "logo.png")

    if os.path.exists(logo_path):
        st.image(logo_path, width=145, output_format="PNG")

    st.markdown(
        '<div class="main-title">InsightForgeAI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Turning Data Into Intelligence, Empowering Tomorrow.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="powered">Powered by Team BlackBox</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Add your dataset")

    st.markdown(
        """
        <div class="card">
            Upload a CSV or Excel dataset to unlock analysis, AI chat,
            forecasting, insights and downloadable reports.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            df = clean_dataframe(load_dataset(uploaded_file))

            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.analysis_name = uploaded_file.name

            st.session_state.history.append(
                {
                    "name": uploaded_file.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "time": datetime.now().strftime("%d %b %Y, %H:%M"),
                }
            )

            st.success(f"Successfully loaded **{uploaded_file.name}**")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Rows", f"{len(df):,}")

            with c2:
                st.metric("Columns", len(df.columns))

            with c3:
                st.metric(
                    "Missing Values",
                    f"{int(df.isnull().sum().sum()):,}",
                )

            with c4:
                st.metric(
                    "Numeric Columns",
                    len(get_numeric_columns(df)),
                )

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)

            st.markdown("### Automatic Insights")
            for insight in generate_insights(df):
                st.markdown("• " + insight)

        except Exception as e:
            st.error(f"Could not read the file: {e}")

    elif st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Current dataset: {st.session_state.file_name}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{len(df):,}")
        with c2:
            st.metric("Columns", len(df.columns))
        with c3:
            st.metric(
                "Missing Values",
                f"{int(df.isnull().sum().sum()):,}",
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    logo_path = os.path.join("assets", "logo.png")

    if os.path.exists(logo_path):
        st.image(logo_path, width=75)

    st.markdown(
        '<div class="sidebar-brand-name">InsightForgeAI</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="workspace-title">Workspace</div>',
        unsafe_allow_html=True,
    )

        # ---------------- WORKSPACE NAVIGATION ----------------

    if "page" not in st.session_state:
        st.session_state.page = "New Analysis"

    workspace_items = [
        ("＋", "New Analysis"),
        ("▥", "Analysis"),
        ("✦", "AI Chat"),
        ("↗", "Forecasting"),
        ("▤", "Reports"),
        ("◷", "History"),
        ("⚙", "Settings"),
    ]

    for icon, name in workspace_items:

        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True
        ):
            st.session_state.page = name
            st.rerun()

    page = st.session_state.page

    st.markdown("---")

    st.markdown(
        '<div class="workspace-title">Current Dataset</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        st.success(st.session_state.file_name)
        st.caption(
            f"{len(st.session_state.df):,} rows × "
            f"{len(st.session_state.df.columns):,} columns"
        )
    else:
        st.info("No dataset uploaded yet.")


# =========================================================
# PAGE ROUTING
# =========================================================

page = st.session_state.page

if page == "New Analysis":
    show_front_page()


elif page == "Analysis":
    st.title("Data Analysis")

    if st.session_state.df is None:
        st.warning("Upload a dataset from New Analysis first.")
    else:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(
            ["Overview", "Statistics", "Charts"]
        )

        with tab1:
            st.subheader("Dataset Overview")
            st.dataframe(
                df.describe(include="all").transpose(),
                use_container_width=True,
            )

            st.subheader("Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if len(missing) == 0:
                st.success("No missing values found.")
            else:
                st.bar_chart(missing)

        with tab2:
            numeric = df.select_dtypes(include=np.number)

            if numeric.empty:
                st.info("No numeric columns were found.")
            else:
                st.dataframe(
                    numeric.describe().transpose(),
                    use_container_width=True,
                )

        with tab3:
            numeric_columns = get_numeric_columns(df)

            if not numeric_columns:
                st.info("No numeric columns available for charting.")
            else:
                selected = st.selectbox(
                    "Select a numeric column",
                    numeric_columns,
                )
                st.line_chart(df[selected])


elif page == "AI Chat":
    st.title("InsightForgeAI Assistant")

    if st.session_state.df is None:
        st.warning("Upload a dataset first from New Analysis.")
    else:
        df = st.session_state.df

        st.caption(
            f"AI is currently analyzing: {st.session_state.file_name}"
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "Ask anything about your dataset..."
        )

        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing your dataset..."):
                    response = ask_ai(prompt, df)

                st.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )


elif page == "Forecasting":
    st.title("Forecasting")

    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state.df
        numeric = get_numeric_columns(df)

        if not numeric:
            st.info("Forecasting requires at least one numeric column.")
        else:
            column = st.selectbox(
                "Select a column to forecast",
                numeric,
            )

            periods = st.slider(
                "Forecast periods",
                min_value=1,
                max_value=30,
                value=7,
            )

            series = df[column].dropna()

            if len(series) < 3:
                st.warning("Not enough data for forecasting.")
            else:
                x = np.arange(len(series))
                coefficients = np.polyfit(
                    x,
                    series.values,
                    1,
                )

                future_x = np.arange(
                    len(series),
                    len(series) + periods,
                )

                predictions = np.polyval(
                    coefficients,
                    future_x,
                )

                forecast_df = pd.DataFrame(
                    {
                        "Period": range(1, periods + 1),
                        "Forecast": predictions,
                    }
                )

                st.dataframe(
                    forecast_df,
                    use_container_width=True,
                )

                st.line_chart(
                    forecast_df.set_index("Period")
                )


elif page == "Reports":
    st.title("Reports")

    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state.df

        st.subheader("Report Preview")

        for insight in generate_insights(df):
            st.markdown("• " + insight)

        st.markdown("---")

        if REPORTLAB_AVAILABLE:
            pdf = create_pdf(df)

            st.download_button(
                "Download PDF Report",
                data=pdf,
                file_name="InsightForgeAI_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.error(
                "PDF support is not installed. Run: pip install reportlab"
            )

        csv_data = df.to_csv(index=False)

        st.download_button(
            "Download Dataset CSV",
            data=csv_data,
            file_name="InsightForgeAI_Dataset.csv",
            mime="text/csv",
        )


elif page == "History":
    st.title("Analysis History")

    if not st.session_state.history:
        st.info("No analyses yet.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(
                f"""
                <div class="card">
                    <h4>{item["name"]}</h4>
                    <p>
                        Rows: {item["rows"]:,} |
                        Columns: {item["columns"]} |
                        Time: {item["time"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "Settings":
    st.title("Settings")

    st.subheader("AI Configuration")

    st.write(
        "Add your OpenAI API key to enable full natural-language AI analysis."
    )

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
    )

    if st.button("Save API Key"):
        st.session_state.api_key = api_key
        st.success("API key saved for this session.")

    st.markdown("---")

    st.subheader("Application")

    st.write(
        f"Current dataset: {st.session_state.file_name or 'None'}"
    )

    if st.button("Clear Current Analysis"):
        st.session_state.df = None
        st.session_state.file_name = None
        st.session_state.messages = []
        st.success("Current analysis cleared.")
        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        InsightForgeAI • Built with ❤️ by Team MOHAMMED AAFRID
    </div>
    """,
    unsafe_allow_html=True,
)
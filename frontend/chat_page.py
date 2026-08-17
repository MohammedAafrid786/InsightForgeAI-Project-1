import streamlit as st
import ollama
import pandas as pd


def build_dataset_context(df):
    """Create a compact but useful description of the uploaded dataset."""

    context = []

    context.append(f"Dataset rows: {len(df)}")
    context.append(f"Dataset columns: {len(df.columns)}")
    context.append("\nCOLUMN INFORMATION:")

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isna().sum())

        context.append(
            f"- {col} | type={dtype} | missing={missing}"
        )

    # Numeric summaries
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        context.append("\nNUMERIC COLUMN SUMMARIES:")

        summary = df[numeric_cols].describe().round(2)

        context.append(summary.to_string())

    # Categorical summaries
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_cols) > 0:
        context.append("\nCATEGORICAL COLUMN SUMMARIES:")

        for col in categorical_cols:
            values = df[col].dropna().value_counts().head(10)

            context.append(
                f"\n{col}:\n{values.to_string()}"
            )

    # Give the AI actual rows to understand the dataset
    context.append("\nSAMPLE DATA:")

    sample = df.head(30).copy()

    context.append(sample.to_string(index=False))

    return "\n".join(context)


def ask_ollama(question, df):

    dataset_context = build_dataset_context(df)

    system_prompt = """
You are InsightForgeAI, a professional business data analyst.

You are analyzing a dataset uploaded by the user.

Rules:
1. Answer questions using the provided dataset information.
2. Never claim you calculated something if the supplied information does not support it.
3. Be concise but useful.
4. Explain important numbers clearly.
5. Use ₹ for Indian currency when appropriate.
6. When comparing categories, regions, products or segments, clearly state the comparison.
7. If the user asks something unrelated to the dataset, answer normally.
8. If the dataset does not contain enough information to answer accurately, say so.
9. Do not mention OpenAI.
10. You are running locally through Ollama and Gemma.
"""

    user_prompt = f"""
Here is the dataset information:

---------------- DATASET ----------------

{dataset_context}

-------------- END DATASET --------------

USER QUESTION:

{question}

Give the best answer possible based on the dataset.
"""

    try:

        response = ollama.chat(
            model="gemma3:latest",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"""
### AI connection error

I couldn't connect to Ollama.

Make sure Ollama is running and that the model exists.

Model:
`gemma3:latest`

Technical error:
`{e}`
"""


def show_chat():

    st.markdown(
        """
        <h1 class="main-title">InsightForgeAI</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="subtitle">Talk to your data</p>
        """,
        unsafe_allow_html=True
    )

    # Make sure a dataset exists
    if "df" not in st.session_state or st.session_state.df is None:

        st.warning(
            "Please upload a CSV or Excel dataset first."
        )

        return

    df = st.session_state.df

    # Dataset status
    st.caption(
        f"AI is analyzing: "
        f"{st.session_state.get('file_name', 'uploaded dataset')}"
    )

    st.success(
        f"Dataset loaded: {len(df):,} rows × {len(df.columns)} columns"
    )

    # Chat history
    if "messages" not in st.session_state:

        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.write(msg["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask anything about your dataset..."
    )

    if prompt:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.write(prompt)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Analyzing your data..."):

                response = ask_ollama(
                    prompt,
                    df
                )

            st.write(response)

        # Save AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
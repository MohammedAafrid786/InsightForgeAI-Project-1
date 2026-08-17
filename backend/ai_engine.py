from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def ask_ai(question, df):

    data_preview = df.head(20).to_string()

    prompt = f"""
You are InsightForgeAI.

Dataset:
{data_preview}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message.content
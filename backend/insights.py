def generate_insights(df):

    insights = []

    insights.append(f"Rows: {df.shape[0]}")
    insights.append(f"Columns: {df.shape[1]}")

    missing = df.isnull().sum().sum()

    insights.append(f"Missing Values: {missing}")

    return insights
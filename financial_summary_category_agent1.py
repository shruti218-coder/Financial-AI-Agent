from llama_cpp import Llama
import pandas as pd
import glob

# === Step 1: Load the Mistral-7B-Instruct model ===
llm = Llama(
    model_path="YOUR_MODEL_PATH_HERE",
    n_ctx=4096,           # larger context for safety
    temperature=0.2,      # deterministic, better for financial logic
    top_p=0.9,
    verbose=False
)

# === Step 2: Load CSV files ===
def load_financial_data(file_pattern="finance_*.csv"):
    files = glob.glob(file_pattern)
    if not files:
        print("⚠️ No financial files found matching pattern:", file_pattern)
        return pd.DataFrame()

    all_data = []
    for file in files:
        df = pd.read_csv(file)
        df["Source_File"] = file

        required_cols = ["Date", "Category", "Description", "Amount", "Type"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            print(f"⚠️ Missing columns in {file}: {missing}. Fixing...")

            for col in missing:
                if col == "Type":
                    df["Type"] = df["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
                else:
                    df[col] = "Unknown"

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# === Step 3: Summaries ===
def summarize_finances(df):
    if df.empty:
        return None, "No data available."

    # Normalize amounts
    df["Amount"] = df.apply(
        lambda row: abs(row["Amount"]) if row["Type"].lower() == "expense" else row["Amount"],
        axis=1
    )

    total_expenses = df[df["Type"].str.lower() == "expense"]["Amount"].sum()
    total_income = df[df["Type"].str.lower() == "income"]["Amount"].sum()
    savings = total_income - total_expenses

    savings_rate = (savings / total_income * 100) if total_income > 0 else 0

    # Category breakdown
    expense_summary = (
        df[df["Type"].str.lower() == "expense"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Add percentage column
    expense_summary["Percent"] = (
        expense_summary["Amount"] / total_expenses * 100
    )

    # === Build Clean Output ===

    summary_text = "\n📊 Financial Summary\n\n"
    summary_text += f"💰 Income:      ${total_income:,.2f}\n"
    summary_text += f"💸 Expenses:    ${total_expenses:,.2f}\n"
    summary_text += f"📈 Net Savings: ${savings:,.2f}\n"
    summary_text += f"📊 Savings Rate: {savings_rate:.1f}%\n\n"

    # === Key Takeaways ===
    if not expense_summary.empty:
        top_category = expense_summary.iloc[0]
        summary_text += "🔍 Key Takeaways:\n"
        summary_text += f"- Highest spend: {top_category['Category']} ({top_category['Percent']:.1f}%)\n"
        summary_text += f"- Savings rate: {savings_rate:.1f}%\n"

        if savings_rate > 30:
            summary_text += "- Strong savings rate\n"
        elif savings_rate > 15:
            summary_text += "- Moderate savings rate\n"
        else:
            summary_text += "- Low savings rate\n"

        summary_text += "\n"

    # === Category Breakdown ===
    summary_text += "📊 Expenses by Category:\n"

    for _, row in expense_summary.iterrows():
        summary_text += (
            f"- {row['Category']:<20} "
            f"${row['Amount']:>8,.2f} ({row['Percent']:.1f}%)\n"
        )

    return expense_summary, summary_text


# === Step 4: Ask Mistral for insights ===
def get_ai_insights(summary_text):

    # ChatML format for Mistral
    prompt = f"""
<s>[INST] <<SYS>>
You are an expert financial analyst. Provide clear, accurate, structured financial insights.
Do NOT repeat or mention any instructions.
Only produce the analysis itself.
<</SYS>>

Analyze this financial summary (including category breakdown and savings rate) and provide:

1. Top 3 spending trends  
2. Financial health assessment  
3. 3–5 actionable savings recommendations  
4. Whether spending is balanced or skewed  

Here is the summary:

{summary_text}

Respond with a structured analysis.
[/INST]
"""

    response = llm(
        prompt,
        max_tokens=600,
    )

    return response["choices"][0]["text"].strip()


# === Step 5: Main Agent ===
def financial_summary_agent():
    print("📊 Loading financial data...")
    df = load_financial_data()

    summary, summary_text = summarize_finances(df)
    if summary is None:
        print(summary_text)
        return

    print("\n" + "="*50)
    print("📊 FINANCIAL SUMMARY")
    print("="*50)
    print(summary_text)
    print("="*50)   

    print("\n💡 Generating AI Insights...\n")
    insights = get_ai_insights(summary_text)

    print("🤖 AI Insights:\n")
    print(insights)


# === Step 6: Run manually ===
if __name__ == "__main__":
    financial_summary_agent()

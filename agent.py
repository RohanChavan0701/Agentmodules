from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from finscraper.scraper_tool.tool import get_stock_info_tool, stock_data_tool, stock_news_tool
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get the OpenAI API key securely
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise EnvironmentError("❌ OPENAI_API_KEY is not set in .env or environment.")

# Prompt template that uses both financial data and news
prompt = PromptTemplate.from_template(
    """You are a seasoned financial analyst.

Your task is to analyze the following financial data and recent news headlines for the company with ticker symbol {ticker}, and recommend whether an investor should **Buy** or **Sell** its stock. Do **not** recommend "Hold".

Use both:
- The company's **financial metrics** (e.g., EPS, revenue, P/E ratio)
- The **market sentiment** from recent news

Then generate a clear recommendation **with justification**.

---

📊 Financial Data:
{financial_data}

📰 News Sentiment:
{news_data}

---

Respond in a {tone} tone. Be concise but insightful.

Your answer must include:
1. 📌 **Your Recommendation** — either "Buy" or "Sell"
2. 💬 **Justification** — explain why, referencing key financials and sentiment
3. 📈 **Confidence Score** — how confident you are (in %)

Format:
Recommendation: <Buy or Sell>  
Justification: <reasoning based on metrics and sentiment>  
Confidence Score: <##%>
"""
)

# Create the LLM
llm = ChatOpenAI(
    openai_api_key=openai_key,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Create the chain using RunnableSequence
chain = prompt | llm


def get_recommendation(ticker: str, tone: str = "formal") -> str:
    # Step 1: Fetch both stock data and news
    financial_data = stock_data_tool.run(ticker)
    news_data = stock_news_tool.run(ticker)

    # Step 2: Call the chain with both
    result = chain.invoke({
        "ticker": ticker,
        "financial_data": financial_data,
        "news_data": news_data,
        "tone": tone
    })

    return result.content


if __name__ == "__main__":
    # Let's analyze multiple stocks with different tones
    stocks_and_tones = [
        ("TSLA", "conversational"),
        ("AAPL", "formal"),
        ("MSFT", "direct")
    ]

    for ticker, tone in stocks_and_tones:
        print(f"\n{'=' * 50}")
        print(f"Analyzing {ticker} in {tone} tone:")
        print(f"{'=' * 50}\n")
        print(get_recommendation(ticker, tone))
        print()



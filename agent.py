from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from finscraper.scraper_tool.tool import get_stock_info_tool, stock_data_tool


# Prompt template that uses the full financial data string
prompt_template = PromptTemplate.from_template(
    """You are a financial analyst. Given the financial data below for the company with ticker symbol {ticker},
provide a Buy/Sell/Hold recommendation and justify your reasoning in a {tone} tone.

Financial Data:
{financial_data}

Recommendation:"""
)

# Create the LLM and the chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

recommendation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)
#code changed


def get_recommendation(ticker: str, tone: str = "formal") -> str:
    from finscraper.scraper_tool.fetcher import fetch_stock_data_yf

    # Step 1: Use the stock data tool
    financial_data = stock_data_tool.run(ticker)

    # Step 2: Pass formatted string into the LLM prompt
    result = recommendation_chain.run({
        "ticker": ticker,
        "financial_data": financial_data,
        "tone": tone
    })

    return result


if __name__ == "__main__":
    ticker = "TSLA"
    tone = "conversational"  # Try "formal", "direct", or "conversational"
    print(get_recommendation(ticker, tone))




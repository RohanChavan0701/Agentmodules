import os
import re
from dotenv import load_dotenv
from agent import get_recommendation
from finscraper.scraper_tool.tool import stock_data_tool
from finscraper.scraper_tool.news_fetcher import fetch_finance_news
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"]

def parse_financials_string(fin_str: str) -> dict:
    lines = fin_str.strip().splitlines()
    result = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip().lower().replace(" ", "_")] = value.strip()
    return result


def verify_recommendation(ticker: str, financial_data: dict, recommendation: str) -> dict:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    try:
        news_articles = fetch_finance_news(ticker)
        headlines = [article["title"] for article in news_articles]
        news_summary = "\n\n".join(
            f"{article['title']} — {article.get('summary', '')}\n{article.get('content', '')[:500]}..."
            for article in news_articles
        )
    except Exception as e:
        news_summary = f"No news available. (Error: {e})"
        headlines = []

    verifier_prompt = PromptTemplate.from_template(
        """..."""  # use your existing detailed prompt here
    )

    verifier_chain = LLMChain(llm=llm, prompt=verifier_prompt)
    financial_string = "\n".join(f"{k}: {v}" for k, v in financial_data.items())

    result = verifier_chain.run({
        "ticker": ticker,
        "financial_data": financial_string,
        "news_summary": news_summary,
        "recommendation": recommendation
    })

    # Extract specific fields from result
    import re
    verdict = re.search(r"Final Verdict:\s*(.*)", result)
    confidence = re.search(r"Confidence Score:\s*(.*)", result)
    headlines_in_justification = re.findall(r"\d+\.\s+(.*)", result)

    return {
        "ticker": ticker,
        "original_recommendation": recommendation,
        "verdict": verdict.group(1).strip() if verdict else "UNCERTAIN",
        "confidence": confidence.group(1).strip() if confidence else "N/A",
        "financials_used": financial_data,
        "news_headlines": headlines[:3],  # raw scraped headlines
        "supporting_headlines_from_llm": headlines_in_justification[:3],
        "justification": result.strip(),
        "raw_llm_output": result.strip()
    }

def run_full_analysis(ticker: str, tone: str = "formal", use_mock: bool = False) -> dict:
    """
    Runs the full Sentimint agent pipeline and returns structured output.
    """
    if use_mock:
        financials = {
            "stock_price": 341.17,
            "pe_ratio": 52.3,
            "eps_ttm": 6.52,
            "market_cap": "1.2T",
            "revenue": "24.7B"
        }
        recommendation = (
            "BUY – Strong EPS, reasonable P/E, and high market cap "
            "make this a solid investment for long-term growth."
        )
    else:
        raw_financials = stock_data_tool.run(ticker)
        financials = parse_financials_string(raw_financials) if isinstance(raw_financials, str) else raw_financials
        recommendation = get_recommendation(ticker, tone)

    verifier_result = verify_recommendation(ticker, financials, recommendation)

    return {
        "ticker": ticker,
        "financials": financials,
        "recommendation": recommendation,
        "verdict": verifier_result.get("verdict"),
        "confidence": verifier_result.get("confidence"),
        "justification": verifier_result.get("justification")
    }

import os
import re
import json
from dotenv import load_dotenv
from agent import get_recommendation
from finscraper.scraper_tool.tool import stock_data_tool, stock_news_tool
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

def apply_heuristic_flags(financials: dict, news_summary: str) -> list:
    flags = []
    try:
        pe_ratio = float(financials.get("pe_ratio", 0))
        if pe_ratio > 100:
            flags.append("🚩 High P/E Ratio (> 100): May indicate overvaluation.")
    except:
        pass

    try:
        revenue = financials.get("revenue", "").replace(",", "").upper()
        if "B" in revenue:
            revenue = float(revenue.replace("B", "")) * 1e9
        elif "M" in revenue:
            revenue = float(revenue.replace("M", "")) * 1e6
        # Simulated check for YoY decline (actual requires historical data)
    except:
        pass

    risky_terms = ["SEC investigation", "fraud", "layoffs"]
    for term in risky_terms:
        if term.lower() in news_summary.lower():
            flags.append(f"🚩 Risk term detected in news: '{term}'")

    return flags

def verify_recommendation(ticker: str, financial_data: dict, recommendation: str) -> dict:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    try:
        news_summary = stock_news_tool.run(ticker)
        headlines = re.findall(r"(.*?)[\n\r]+", news_summary)[:5]
    except Exception as e:
        news_summary = f"No news available. (Error: {e})"
        headlines = []

    try:
        financial_data_str = stock_data_tool.run(ticker)
    except Exception as e:
        financial_data_str = f"Unable to fetch financial data. (Error: {e})"

    heuristic_flags = apply_heuristic_flags(financial_data, news_summary)

    verifier_prompt = PromptTemplate.from_template(
        """
You are the final verification agent responsible for validating an investment recommendation for the company {ticker}.

Inputs:
- News Coverage:
{news_summary}

- Financial Data:
{financial_data}

- Original Recommendation:
"{recommendation}"

Task:
1. Evaluate if the recommendation aligns with the news and financials.
2. Highlight 2–3 relevant news items supporting or contradicting it.
3. Assess if financial metrics strengthen or weaken the recommendation.
4. Adjust recommendation if necessary.
5. Return a final verdict and confidence score (0–100%).

Output format:
- Final Verdict: [BUY / SELL / HOLD / revised recommendation]
- Key Headlines Supporting Verdict:
  1. ...
  2. ...
  3. ...
- Financial Validation: ...
- Confidence Score: ...
"""
    )

    verifier_chain = LLMChain(llm=llm, prompt=verifier_prompt)
    result = verifier_chain.run({
        "ticker": ticker,
        "financial_data": financial_data_str,
        "news_summary": news_summary,
        "recommendation": recommendation
    })

    verdict = re.search(r"Final Verdict:\s*(.*)", result)
    confidence = re.search(r"Confidence Score:\s*(.*)", result)
    justification = result.strip()
    extracted_support = re.findall(r"\d\.\s+(.*)", justification)

    return {
        "ticker": ticker,
        "original_recommendation": recommendation,
        "verdict": verdict.group(1).strip() if verdict else "UNCERTAIN",
        "confidence": confidence.group(1).strip() if confidence else "N/A",
        "justification": justification,
        "supporting_headlines_from_llm": extracted_support[:3],
        "news_headlines": headlines[:3],
        "financials_used": financial_data_str,
        "heuristic_flags": heuristic_flags,
        "raw_llm_output": result.strip()
    }

def run_full_analysis(ticker: str, tone: str = "formal", use_mock: bool = False) -> dict:
    if use_mock:
        financials = {
            "stock_price": 341.17,
            "pe_ratio": 152.3,
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
        "justification": verifier_result.get("justification"),
        "supporting_headlines": verifier_result.get("supporting_headlines_from_llm"),
        "news_headlines": verifier_result.get("news_headlines"),
        "heuristic_flags": verifier_result.get("heuristic_flags", [])
    }

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., TSLA): ").upper()
    output = run_full_analysis(ticker=ticker, tone="formal", use_mock=False)

    import json
    print("\n📦 FULL OUTPUT:")
    print(json.dumps(output, indent=2))

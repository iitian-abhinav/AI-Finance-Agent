BENCHMARK_CASES = [
    {
        "case_id": "sec_aapl_001", "category": "sec_filing", "ticker": "AAPL",
        "prompt": "Analyze Apple Inc. (AAPL) using the latest SEC filing data available to the system. Focus on the latest 10-K, latest 10-Q, and recent 8-K filings. Clearly distinguish retrieved facts from interpretation and do not infer unavailable contents.",
        "required_terms": ["10-K", "10-Q"],
        "forbidden_phrases": ["although not explicitly stated", "historically, Apple"]
    },
    {
        "case_id": "financial_aapl_001", "category": "financial_statement", "ticker": "AAPL",
        "prompt": "Perform a financial statement analysis of Apple Inc. (AAPL) using only SEC XBRL financial data and Python-calculated ratios supplied by the system. Analyze revenue, earnings, profitability, liquidity, leverage, operating cash flow, capital expenditure, and free cash flow where supported.",
        "required_terms": ["revenue", "net income", "cash flow"],
        "forbidden_phrases": ["although not explicitly stated", "historically, Apple"]
    },
    {
        "case_id": "market_aapl_001", "category": "market_data", "ticker": "AAPL",
        "prompt": "Provide a technical and market-data analysis of Apple Inc. (AAPL) using only market data and technical indicators retrieved and calculated by the system. Analyze returns, moving averages, RSI, MACD, historical volatility, maximum drawdown, and the 52-week range.",
        "required_terms": ["RSI", "MACD"],
        "forbidden_phrases": ["guaranteed", "will continue to rise", "has room to grow"]
    },
    {
        "case_id": "news_aapl_001", "category": "news", "ticker": "AAPL",
        "prompt": "Analyze recent news retrieved by the system for Apple Inc. (AAPL). Group directly relevant articles into themes, separate confirmed developments from interpretation, and classify overall news flow.",
        "required_terms": ["news"],
        "forbidden_phrases": ["although not directly mentioned", "historically, Apple"]
    },
    {
        "case_id": "comparative_aapl_001", "category": "comparative", "ticker": "AAPL",
        "prompt": "Perform a comparative analysis of Apple Inc. (AAPL) using only peer companies and peer metrics supplied by the system. Compare only metrics available for both Apple and retrieved peers. Provide an evidence-grounded SWOT analysis.",
        "required_terms": ["peer"],
        "forbidden_phrases": ["growing demand for technology", "expansion into new markets", "intense competition"]
    },
    {
        "case_id": "investment_aapl_001", "category": "investment_decision", "ticker": "AAPL",
        "prompt": "Evaluate Apple Inc. (AAPL) as an investment using only evidence supplied by the system. Produce a bull case, bear case, catalysts, risks, uncertainty and missing data, and a final BUY, HOLD, SELL or AVOID rating. Provide a price target only if sufficient valuation inputs are available.",
        "required_terms": ["bull case", "bear case", "rating"],
        "forbidden_phrases": ["although not explicitly stated", "though not directly mentioned", "historically, Apple", "brand strength and pricing power"]
    }
]

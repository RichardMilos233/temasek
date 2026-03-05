import yfinance as yf
import pandas as pd
import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset

def setup_accelerated_pipeline():
    print("Detecting hardware...")
    
    # 1. Intelligent Hardware Routing
    if torch.cuda.is_available():
        # Windows + NVIDIA 4060
        device = torch.device("cuda:0")
        print(f"🚀 Hardware Acceleration: NVIDIA GPU ({torch.cuda.get_device_name(0)}) detected.")
    elif torch.backends.mps.is_available():
        # Mac + M1 (Apple Silicon)
        device = torch.device("mps")
        print("🚀 Hardware Acceleration: Apple M1 (MPS) detected.")
    else:
        # Fallback
        device = torch.device("cpu")
        print("⚠️ No GPU detected. Falling back to CPU.")

    print("Loading models into VRAM/Unified Memory...")

    # 2. ESG Topic Classifier
    esg_model_name = "yiyanghkust/finbert-esg"
    esg_nlp = pipeline("text-classification", model=esg_model_name, device=device)
    
    # 3. Financial Sentiment Classifier (Explicitly loaded to bypass the config bug)
    tone_model_name = "yiyanghkust/finbert-tone"
    tone_tokenizer = BertTokenizer.from_pretrained(tone_model_name)
    tone_model = BertForSequenceClassification.from_pretrained(tone_model_name, num_labels=3)
    tone_nlp = pipeline("text-classification", model=tone_model, tokenizer=tone_tokenizer, device=device)
    
    return esg_nlp, tone_nlp    

def fetch_ticker_news(ticker_symbol):
    print(f"Fetching recent news for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    
    # Extract titles and publishers into a DataFrame
    data = []
    for article in news:
        # print(article.get("content", "").get("title", ""))
        data.append({
            "ticker": ticker_symbol,
            "publisher": article.get("publisher", "Unknown"),
            "text": article.get("content", "").get("title", "")
        })
    # print(data)
    
    return pd.DataFrame(data)

def analyze_esg_data_in_batches(df, esg_nlp, tone_nlp, batch_size=16):
    print(f"Processing {len(df)} records in batches of {batch_size}...")
    
    texts = df['text'].tolist()
    
    # Run ESG classification in batches
    esg_results = esg_nlp(texts, batch_size=batch_size)
    df['esg_category'] = [res['label'] for res in esg_results]
    
    # Filter for texts that are actually ESG-related to save compute time
    esg_mask = df['esg_category'] != "None"
    esg_texts = df.loc[esg_mask, 'text'].tolist()
    
    # Run Sentiment analysis only on the ESG-related text
    df['sentiment'] = "None"
    df['sentiment_score'] = 0.0
    
    if esg_texts:
        tone_results = tone_nlp(esg_texts, batch_size=batch_size)
        
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        
        # Map results back to the DataFrame
        df.loc[esg_mask, 'sentiment'] = [res['label'] for res in tone_results]
        df.loc[esg_mask, 'sentiment_score'] = [sentiment_map[res['label']] for res in tone_results]

    return df

if __name__ == "__main__":
    # 1. Setup the accelerated NLP pipeline
    esg_nlp, tone_nlp = setup_accelerated_pipeline()
    
    # 2. Retrieve real data (Example: Sourcing news for a few tickers)
    tickers = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "IBM", "ORCL",
        # Energy
        "XOM", "CVX", "BP", "SHEL", "TTE", "COP", "SLB", "EOG", "MPC", "PSX",
        # Finance
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK",
        # Consumer & Retail
        "WMT", "TGT", "COST", "HD", "MCD", "SBUX", "NKE", "PG", "KO", "PEP",
        # Healthcare, Auto, & Industrials
        "JNJ", "PFE", "UNH", "TSLA", "F", "GM", "BA", "LMT", "CAT", "DE"
    ]
    # tickers = ["AAPL"]
    dfs = [fetch_ticker_news(t) for t in tickers]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Process the data efficiently
    if not combined_df.empty:
        results_df = analyze_esg_data_in_batches(combined_df, esg_nlp, tone_nlp, batch_size=8)
        
        print("\n" + "="*80)
        print("SAMPLE OUTPUT:")
        # Display only articles that triggered an ESG signal
        signals_only = results_df[results_df['esg_category'] != "None"]
        if not signals_only.empty:
            print(signals_only[['ticker', 'text', 'esg_category', 'sentiment', 'sentiment_score']].head())
        else:
            print("No significant ESG signals detected in the latest headlines.")
        print("="*80)
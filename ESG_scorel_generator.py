from transformers import pipeline, BertTokenizer, BertForSequenceClassification

def setup_esg_pipeline():
    print("Loading models... ")
    
    # 1. Load the ESG Topic Classifier (Auto-detect works fine for this one)
    esg_model_name = "yiyanghkust/finbert-esg"
    esg_nlp = pipeline("text-classification", model=esg_model_name, tokenizer=esg_model_name)
    
    # 2. Load the Financial Sentiment Classifier (Explicitly loaded to bypass the bug)
    tone_model_name = "yiyanghkust/finbert-tone"
    
    # We explicitly tell it to use the BERT architecture and tokenizer
    tone_tokenizer = BertTokenizer.from_pretrained(tone_model_name)
    tone_model = BertForSequenceClassification.from_pretrained(tone_model_name, num_labels=3)
    
    # Pass the pre-loaded model and tokenizer into the pipeline
    tone_nlp = pipeline("text-classification", model=tone_model, tokenizer=tone_tokenizer)
    
    return esg_nlp, tone_nlp

def calculate_esg_score(texts, esg_nlp, tone_nlp):
    # Initialize our scoring buckets
    esg_scores = {"Environmental": 0, "Social": 0, "Governance": 0}
    
    # Simple mapping for sentiment to numerical value
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    
    print("-" * 60)
    for text in texts:
        # Step A: Identify the ESG category
        esg_result = esg_nlp(text)[0]
        category = esg_result['label']
        
        # Step B: If it's an ESG topic, determine the sentiment
        if category in esg_scores:
            tone_result = tone_nlp(text)[0]
            sentiment = tone_result['label']
            weight = sentiment_map[sentiment]
            
            # Step C: Aggregate the score
            esg_scores[category] += weight
            
            print(f"Text:      '{text}'")
            print(f"Extracted: [{category}] with [{sentiment}] sentiment -> Score Change: {weight}\n")
        else:
            print(f"Text:      '{text}'")
            print("Extracted: [None] -> Not an ESG topic. Ignored.\n")
            
    return esg_scores

if __name__ == "__main__":
    esg_nlp, tone_nlp = setup_esg_pipeline()
    
    # Simulate a stream of unstructured text (e.g., news headlines or report sentences)
    sample_texts = [
        "The company cut its carbon footprint by 20% this year using renewable energy.",
        "Workers are striking over unfair labor practices and poor safety conditions.",
        "The board of directors lacks diversity and independent oversight.",
        "Revenue increased by 15% in Q3 due to strong sales in Europe.", # Financial noise
        "They donated $5 million to local education initiatives in the community."
    ]
    
    final_scores = calculate_esg_score(sample_texts, esg_nlp, tone_nlp)
    
    print("-" * 60)
    print("Final Aggregated ESG Sentiment Scores:")
    for category, score in final_scores.items():
        print(f"{category}: {score}")
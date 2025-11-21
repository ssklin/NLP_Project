import json
import os
import glob
from keyword_utils import load_nlp_model, get_sentiment_label, extract_aspect_keywords, ASPECTS

from constants import OUTPUT_DIR

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def mount_google_drive():
    """
    Attempts to mount Google Drive if running in Google Colab.
    Returns the path to the data folder.
    """
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        return '/content/drive/MyDrive/data' 
    except ImportError:
        print("Not running in Google Colab or drive module not found.")
        return './data' # Default local path

def load_reviews(data_dir):
    """
    Loads all JSON files from the specified directory.
    """
    json_files = glob.glob(os.path.join(data_dir, "*_reviews.json"))
    all_reviews = {}
    
    print(f"Looking for files in: {data_dir}")
    if not json_files:
        print("No JSON files found! Please check the data directory path.")
        return {}

    for file_path in json_files:
        restaurant_name = os.path.basename(file_path).replace("_reviews.json", "")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
                all_reviews[restaurant_name] = reviews
            print(f"Loaded {len(reviews)} reviews for {restaurant_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_reviews

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    # 1. Setup
    data_path = mount_google_drive()
    # Fallback to local './data' if the drive path doesn't exist
    if not os.path.exists(data_path):
        print(f"Path {data_path} does not exist. Using './data' instead.")
        data_path = './data'

    nlp = load_nlp_model()
    if not nlp:
        return

    # 2. Load Data
    reviews_data = load_reviews(data_path)
    
    # 3. Process Reviews & Save Separately
    print("\nStarting Keyword Extraction...")
    
    for restaurant, reviews in reviews_data.items():
        print(f"Processing {restaurant}...")
        
        # Structure for better readability: Group by Aspect -> Sentiment
        restaurant_output = {
            "Restaurant": restaurant,
            "Total_Reviews": len(reviews),
            "Aspect_Analysis": {
                aspect: {"Positive": [], "Negative": [], "Neutral": []} 
                for aspect in ASPECTS.keys()
            }
        }
        
        keyword_count = 0
        
        for review in reviews:
            text = review.get('text', '')
            stars = review.get('stars', '')
            
            if not text:
                continue
                
            sentiment = get_sentiment_label(stars)
            
            # Extract keywords
            aspect_keywords = extract_aspect_keywords(text, nlp)
            
            if aspect_keywords:
                for aspect, keywords in aspect_keywords.items():
                    for kw in keywords:
                        # Add to the structured output
                        entry = {
                            "Keyword": kw,
                            "Context": text
                        }
                        restaurant_output["Aspect_Analysis"][aspect][sentiment].append(entry)
                        keyword_count += 1

        # 4. Save Individual Result
        # Sanitize filename
        safe_name = "".join([c for c in restaurant if c.isalpha() or c.isdigit() or c==' ']).strip()
        output_file = os.path.join(OUTPUT_DIR, f"{safe_name}_keywords.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(restaurant_output, f, indent=4, ensure_ascii=False)
            
        print(f"  -> Saved {keyword_count} keywords to {output_file}")

    print(f"\nExtraction complete! Check the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()

import spacy
from collections import defaultdict

# Define Aspect Keywords (Seeds)
ASPECTS = {
    "Taste": [
        "food", "flavor", "taste", "delicious", "tasty", "yummy", "bland", "salty", 
        "sweet", "fresh", "chicken", "pizza", "burger", "sandwich", "steak", "meat",
        "sauce", "seasoning", "spicy", "hot", "cold", "drink", "beverage", "menu",
        "dish", "meal", "breakfast", "lunch", "dinner", "portion"
    ],
    "Price": [
        "price", "cost", "expensive", "cheap", "value", "worth", "dollar", "bill", 
        "overpriced", "affordable", "reasonable", "check"
    ],
    "Environment": [
        "atmosphere", "environment", "vibe", "clean", "dirty", "noisy", "quiet", 
        "decor", "seating", "space", "music", "lighting", "view", "interior", 
        "bathroom", "restroom", "table", "chair"
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "manager", "friendly", "rude", 
        "helpful", "attentive", "polite", "server", "host", "hostess", "bartender"
    ],
    "Waiting Time": [
        "wait", "time", "minute", "hour", "queue", "line", "slow", "fast", "quick",
        "rush", "busy", "delay", "reservation"
    ]
}

def load_nlp_model():
    """
    Loads the Spacy English model.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Spacy model not found. run: python -m spacy download en_core_web_sm and try again.")
        return None

def get_sentiment_label(stars_str):
    """
    Converts star string (e.g., "5 stars") to sentiment label.
    """
    try:
        stars = int(stars_str.split()[0])
        if stars > 3:
            return "Positive"
        elif stars < 3:
            return "Negative"
        else: # == 3
            return "Neutral"
    except (ValueError, IndexError):
        return "Neutral"

def extract_aspect_keywords(text, nlp):
    """
    Extracts keywords/phrases related to aspects from the text using Spacy.
    Returns a dictionary of {Aspect: [keywords/phrases]}.
    """
    doc = nlp(text)
    extracted = defaultdict(list)
    
    aspect_map = {}
    for aspect, keywords in ASPECTS.items():
        for kw in keywords:
            aspect_map[kw.lower()] = aspect

    for token in doc:
        # Check if the token lemma is a seed keyword
        if token.lemma_.lower() in aspect_map:
            aspect = aspect_map[token.lemma_.lower()]
            
            # 1. Adjective + Noun (e.g., "good food")
            # If token is a noun, look for adjective children
            if token.pos_ in ["NOUN", "PROPN"]:
                for child in token.children:
                    if child.pos_ == "ADJ":
                        phrase = f"{child.text} {token.text}"
                        extracted[aspect].append(phrase)
            
            # 2. Noun + Verb (e.g., "service was slow")
            # If token is a subject, look for head verb + attribute
            if token.dep_ == "nsubj":
                verb = token.head
                for child in verb.children:
                    if child.pos_ == "ADJ" and child.dep_ == "acomp":
                        phrase = f"{token.text} {verb.text} {child.text}"
                        extracted[aspect].append(phrase)

    return extracted

def format_keywords_for_training(extracted_keywords):
    """
    Formats the extracted keywords dictionary into a string for training.
    Format: "Aspect: kw1, kw2 | Aspect: kw3"
    """
    if not extracted_keywords:
        return None

    parts = []
    for aspect, kws in extracted_keywords.items():
        unique_kws = list(set(kws))
        if unique_kws:
            parts.append(f"{aspect}: {', '.join(unique_kws)}")
    
    return " | ".join(parts)

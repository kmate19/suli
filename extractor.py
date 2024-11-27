import re
from collections import Counter

def extract_features(email):
    features = {}
    words = email.split()
    total_words = len(words)

    subject = email.split("\r\n")[0].split(" ")[1:]

    try:
        time = re.findall(r"on \d{2} \/\s\d{2} \/\s\d{4}\s+\d{2}\s:\s\d{2}\s(?:am)", email)[0]
        if "\r\n" in time:
            time = int(time.split("\r\n")[1][0:2])
        else:
            time = int(time.split(" ")[6])
    except: 
        time = 12

    try: # idk
        to = email.split("\r\nto : ")[1]

        if "\r\ncc : " in to:
            to = to.split("\r\ncc : ")[0].split(" , ")
        if "\r\ncc :" in to:
            to = to.split("\r\ncc :")[0].split(" , ")
        else:
            raise
    except:
        to = []

    # Feature calculations
    features["word_repetition"] = sum(v > 1 for v in Counter(words).values())
    features["keyword_match"] = sum(1 for word in words if word.lower() in subject)
    features["length"] = total_words
    features["multiple_recipients"] = len(to) > 1
    features["special_characters"] = len(re.findall(r"[!@#$%^&*]", email))
    features["time_of_sending"] = 1 if time < 6 else 0
    # dataset never has the below features
    features["has_html"] = "<html>" in email.lower()
    features["links"] = len(re.findall(r"https?://", email))
    features["uppercase_ratio"] = sum(1 for word in words if word.isupper()) / total_words
    features["domain_reputation"] = 0
    
    return features

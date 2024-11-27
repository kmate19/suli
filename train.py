from collections import Counter
from extractor import extract_features


def train_naive_bayes(training_data):
    print(f"training_data {training_data}")
    classes = set(label for label in training_data.get("label"))
    print(f"classes {classes}")
    word_probs = {cls: {} for cls in classes}
    class_priors = {cls: 0.0 for cls in classes}

    for cls in classes:
        class_emails = training_data[training_data['label'] == cls]['text']
        print(f"class_emails {len(class_emails)} {cls}")
        class_priors[cls] = len(class_emails) / len(training_data)
        print(f"class_priors {class_priors[cls]} {cls}")

        feature_totals = Counter()
        for email in class_emails:
            features = extract_features(email)
            feature_totals.update(features)

        total_count = sum(feature_totals.values())
        print(f"total_count {total_count}")

        for feature, count in feature_totals.items():
            word_probs[cls][feature] = (count + 1) / (total_count + len(feature_totals))  # laplace
    
    return word_probs, class_priors


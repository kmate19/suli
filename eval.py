from extractor import extract_features
import numpy as np

def predict(email, word_probs, class_priors):
    features = extract_features(email)
    max_prob = -np.inf
    predicted_class = None

    for cls, prior in class_priors.items():
        log_prob = np.log(prior)
        for feature, value in features.items():
            prob = word_probs[cls].get(feature, 1 / len(features))
            log_prob += np.log(prob) * value
        
        if log_prob > max_prob:
            max_prob = log_prob
            predicted_class = cls

    return predicted_class

def evaluate(test_data, word_probs, class_priors):
    correct = 0
    for row in test_data.itertuples(index=False):
        _,label,text,_ = row
        if predict(text, word_probs, class_priors) == label:
            correct += 1
    return correct / len(test_data)


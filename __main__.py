from train import train_naive_bayes
from eval import evaluate
from correlation import calculate_correlation
import pandas as pd

# dataset used: https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data

training_data = pd.read_csv("spam_ham_dataset.csv")

word_probs, class_priors = train_naive_bayes(training_data)

test_data = pd.read_csv("spam_ham_dataset.csv")

print(word_probs)
print("Asd")
print(class_priors)
print('\n')

accuracy = evaluate(test_data, word_probs, class_priors)
print(f"accuracy: {accuracy * 100}%")

correlation_matrix = calculate_correlation(training_data + test_data)
print(correlation_matrix)

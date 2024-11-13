import pandas as pd
import re

# Load the raw datasets
train_data = pd.read_csv('path_to_your_raw_train_data.tsv', sep='\t')
test_data = pd.read_csv('path_to_your_raw_test_data.tsv', sep='\t')

# Basic stop words list (since we cannot download NLTK's list in this environment)
basic_stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", 
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now"
}

# Function to preprocess text
def basic_preprocess_review(review_text):
    # Convert to lowercase
    review_text = review_text.lower()
    # Remove special characters and numbers
    review_text = re.sub(r"[^a-z\s]", "", review_text)
    # Tokenize and remove stop words
    words = review_text.split()
    words = [word for word in words if word not in basic_stop_words]
    # Join back into a single string
    return " ".join(words)

# Cleaning the datasets
def clean_data(data):
    # Drop unnecessary column
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')
    # Fill missing values in the 'condition' column
    data['condition'].fillna('Unknown', inplace=True)
    # Apply text preprocessing to the 'review' column
    data['review'] = data['review'].apply(basic_preprocess_review)
    return data

# Apply cleaning function to both training and test datasets
train_data_cleaned = clean_data(train_data)
test_data_cleaned = clean_data(test_data)

# Save the cleaned data to new CSV files
train_data_cleaned.to_csv('drugsComTrain_cleaned.csv', index=False)
test_data_cleaned.to_csv('drugsComTest_cleaned.csv', index=False)

print("Data cleaning complete. Cleaned files saved as 'drugsComTrain_cleaned.csv' and 'drugsComTest_cleaned.csv'")

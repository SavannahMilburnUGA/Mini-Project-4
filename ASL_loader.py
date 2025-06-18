# Load ASL fingerspelling dataset from Kaggle
# sign_language_loader.py
# explore_dataset.py
import kagglehub
import os
import pandas as pd

# Download and explore
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Path to dataset files:", path)

print("Files in dataset:")
for file in os.listdir(path):
    print(file)

# Examine CSV files 
import pandas as pd

# Look at the training data
train_df = pd.read_csv(os.path.join(path, 'sign_mnist_train.csv'))
print("\n=== TRAINING DATA ===")
print("Shape:", train_df.shape)
print("Column names (first 10):", list(train_df.columns[:10]))
print("Column names (last 5):", list(train_df.columns[-5:]))
print("First few rows (first 10 columns):")
print(train_df.iloc[:5, :10])

# Look at the test data
test_df = pd.read_csv(os.path.join(path, 'sign_mnist_test.csv'))
print("\n=== TEST DATA ===")
print("Shape:", test_df.shape)
print("Labels distribution in training:")
print(train_df['label'].value_counts().sort_index())
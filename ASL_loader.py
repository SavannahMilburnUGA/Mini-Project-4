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
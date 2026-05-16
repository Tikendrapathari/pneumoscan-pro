import pandas as pd
import os

base_path = r"E:/dataset/lungopacity"
labels_path = os.path.join(base_path, "stage_2_train_labels.csv")

df = pd.read_csv(labels_path)
print("="*50)
print("CSV FILE STRUCTURE")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nTarget distribution:\n{df.iloc[:,1].value_counts() if df.shape[1]>1 else 'Only one column'}")
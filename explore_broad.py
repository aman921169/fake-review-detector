import pandas as pd
 
df = pd.read_csv("fake reviews dataset.csv")
 
print("Shape:", df.shape)
print()
print("Columns:", df.columns.tolist())
print()
print("First 3 rows:")
print(df.head(3))
print()
print("Label counts:")
print(df["label"].value_counts())
print()
print("Category counts:")
print(df["category"].value_counts())
 
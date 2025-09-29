# csc172-data-cleaning-bansao
# Data Cleaning with AI Support

## Student Information
- Name: HUSSAM M. BANSAO
- Course Year: BSCS 4
- Date: 2025-09-29

## Dataset
- Source: Kaggle — Titanic dataset  
  https://www.kaggle.com/c/titanic/data
- Name: Titanic — Machine Learning from Disaster

## Why this dataset
The Titanic dataset is beginner-friendly and contains typical cleaning problems: missing values (Age, Cabin, Embarked), inconsistent categorical formatting, and some outliers in numeric fields like Fare.

## Issues found 
- Missing values:
  - `Age`: many NaNs
  - `Cabin`: many NaNs (sparse)
  - `Embarked`: a few NaNs
- Duplicates: possible identical passenger rows (we check & drop)
- Inconsistencies:
  - Name strings with extra spaces
  - Ticket/Cabin formats inconsistent
- Outliers:
  - `Fare` has a heavy right tail (high fares)
  - `Age` extremes (check for implausible ages)

## Cleaning steps (high level)
1. Load raw CSV `data/raw_dataset.csv`.
2. Exploratory checks: `df.info()`, `df.describe()`, missing value counts, duplicates, sample rows.
3. Handle missing values:
   - Fill `Age` with median (group-based median by Pclass + Sex also provided as optional improved strategy).
   - Fill `Embarked` with mode.
   - For `Cabin`, create `HasCabin` boolean; optionally extract deck letter from non-null cabin entries; otherwise drop `Cabin`.
4. Remove duplicates with `df.drop_duplicates()`.
5. Standardize formats:
   - `Name` trimmed, `Sex` lowercased, `Embarked` uppercased.
   - Convert `Pclass` to categorical.
6. Outlier detection & treatment:
   - Use IQR method on `Fare` and `Age` to optionally remove or cap extreme outliers (demonstrated both approaches).
7. Save cleaned dataset to `data/cleaned_dataset.csv`.

## AI prompts used
- Prompt 1:
"Generate a reproducible Pandas notebook (Python) that loads the Kaggle Titanic dataset,
performs exploration (info, describe, missing values), handles missing Age, Cabin, and Embarked,
removes duplicates, standardizes formats for Name/Sex/Embarked, detects outliers using IQR,
and saves the cleaned CSV. Include before/after snapshots for shape and sample rows and include comments."

- Generated code (snippet from the AI assistant I used — full code is included in `notebooks/data_cleaning.ipynb`):
```python
import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw_dataset.csv")
print("Before shape:", df.shape)
print(df.info())

# Missing value handling
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['HasCabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin'])  # optional

# Duplicates
df = df.drop_duplicates()

# Standardize
df['Name'] = df['Name'].str.strip()
df['Sex'] = df['Sex'].str.lower()
df['Embarked'] = df['Embarked'].str.upper()

# Outlier detection (IQR for Fare)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df['Fare'] < (Q1 - 1.5*IQR)) | (df['Fare'] > (Q3 + 1.5*IQR)))
df_filtered = df[mask]

df_filtered.to_csv("../data/cleaned_dataset.csv", index=False)

Results:

Rows before: 891
Rows after: 891
Columns before: 12
Columns after: 14

How I used an AI tool:
I used an AI assistant to generate a reproducible Pandas notebook and to suggest robust approaches for filling Age (median vs. Pclass/Sex-based median), for extracting cabin decks, and for handling Fare outliers. The prompt and generated code snippet are above. I edited the generated output for clarity and reproducibility, and included comments and before/after snapshot cells in the notebook.

Video

Link: YOUR_VIDEO_LINK_HERE (YouTube unlisted or Google Drive)
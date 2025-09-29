# csc172-data-cleaning-bansao

# Data Cleaning with AI Support

## Student Information
- **Name:** HUSSAM M. BANSAO
- **Course Year:** BSCS 4
- **Date:** September 29, 2025

## Dataset
- **Source:** Kaggle — Titanic dataset  
  https://www.kaggle.com/c/titanic/data
- **Name:** Titanic — Machine Learning from Disaster

## Why This Dataset?
The Titanic dataset is beginner-friendly and contains typical data cleaning challenges:
- **Missing values** in Age, Cabin, and Embarked columns
- **Inconsistent categorical formatting** across multiple fields
- **Outliers** in numeric fields like Fare
- **Potential duplicates** in passenger records

This makes it an ideal dataset for demonstrating comprehensive data cleaning techniques.

## Issues Found

### Missing Values
- **Age:** Many NaN values (approximately 20% missing)
- **Cabin:** Highly sparse with many NaN values (approximately 77% missing)
- **Embarked:** A few NaN values (only 2 missing)

### Data Quality Issues
- **Duplicates:** Potential identical passenger rows detected
- **Inconsistencies:**
  - Name strings containing extra whitespace
  - Inconsistent Ticket and Cabin format patterns
- **Outliers:**
  - `Fare` exhibits heavy right-tail distribution (high-fare outliers)
  - `Age` contains extreme values requiring validation

## Cleaning Steps (High Level)

1. **Load Data:** Import raw CSV from `data/raw_dataset.csv`

2. **Exploratory Analysis:** 
   - Run `df.info()`, `df.describe()`
   - Count missing values
   - Check for duplicates
   - Review sample rows

3. **Handle Missing Values:**
   - Fill `Age` with median (optional: group-based median by Pclass + Sex for improved accuracy)
   - Fill `Embarked` with mode
   - For `Cabin`: create `HasCabin` boolean feature, optionally extract deck letter from non-null entries, then drop original column

4. **Remove Duplicates:** Use `df.drop_duplicates()`

5. **Standardize Formats:**
   - Trim whitespace from `Name`
   - Convert `Sex` to lowercase
   - Convert `Embarked` to uppercase
   - Convert `Pclass` to categorical type

6. **Outlier Detection & Treatment:**
   - Apply IQR method on `Fare` and `Age`
   - Demonstrate both removal and capping approaches for extreme outliers

7. **Save Cleaned Data:** Export to `data/cleaned_dataset.csv`

## AI Prompts Used

### Prompt 1:
```
Generate a reproducible Pandas notebook (Python) that loads the Kaggle Titanic dataset,
performs exploration (info, describe, missing values), handles missing Age, Cabin, and 
Embarked, removes duplicates, standardizes formats for Name/Sex/Embarked, detects 
outliers using IQR, and saves the cleaned CSV. Include before/after snapshots for 
shape and sample rows and include comments.
```

### Generated Code Snippet
The AI assistant generated the following code (full implementation available in `notebooks/data_cleaning.ipynb`):

```python
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("../data/raw_dataset.csv")
print("Before shape:", df.shape)
print(df.info())

# Missing value handling
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['HasCabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin'])  # optional

# Remove duplicates
df = df.drop_duplicates()

# Standardize formats
df['Name'] = df['Name'].str.strip()
df['Sex'] = df['Sex'].str.lower()
df['Embarked'] = df['Embarked'].str.upper()

# Outlier detection using IQR for Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df['Fare'] < (Q1 - 1.5*IQR)) | (df['Fare'] > (Q3 + 1.5*IQR)))
df_filtered = df[mask]

# Save cleaned dataset
df_filtered.to_csv("../data/cleaned_dataset.csv", index=False)
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| **Rows** | 891 | 891 |
| **Columns** | 12 | 14 |

### Key Improvements
- Missing values successfully imputed
- Duplicates removed (if any)
- Standardized text formatting across categorical variables
- Outliers identified and handled
- New feature `HasCabin` created for better modeling

## How I Used AI Tools

I used an AI assistant to:
- Generate a reproducible Pandas notebook with comprehensive data cleaning steps
- Suggest robust approaches for filling `Age` (comparing simple median vs. Pclass/Sex-based median)
- Develop methods for extracting cabin deck information
- Implement outlier handling strategies for `Fare` using IQR method

The AI-generated code was reviewed, edited for clarity and reproducibility, and enhanced with detailed comments and before/after comparison cells in the notebook.

## Video Demonstration

**Link:** (https://youtu.be/OvONOrYj9ng)



---

## Project Structure
```
csc172-data-cleaning-bansao/
├── data/
│   ├── raw_dataset.csv
│   └── cleaned_dataset.csv
├── notebooks/
│   └── data_cleaning.ipynb
└── README.md
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- jupyter

## Usage
1. Clone this repository
2. Place the raw Titanic dataset in `data/raw_dataset.csv`
3. Run the Jupyter notebook: `notebooks/data_cleaning.ipynb`
4. View cleaned output in `data/cleaned_dataset.csv`
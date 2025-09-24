## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
```
<img width="866" height="153" alt="image" src="https://github.com/user-attachments/assets/211cfd48-1bcc-4f7e-ba65-044e6ebc98ce" />

# Load datasets
```
df1 = pd.read_csv("/content/Encoding Data.csv")
df2 = pd.read_csv("/content/Data_to_Transform.csv")
df3 = pd.read_csv("/content/data.csv")
```
<img width="720" height="127" alt="image" src="https://github.com/user-attachments/assets/92bfaf29-2079-4b00-a100-f5ab16a5b520" />

# Merge datasets
```
merged_df = df1.merge(df3, on=["id", "bin_1", "bin_2"], how="inner")
merged_df = merged_df.merge(df2, left_index=True, right_index=True, how="inner")
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # remove duplicates
```
<img width="951" height="129" alt="image" src="https://github.com/user-attachments/assets/57e7ca39-8312-4237-96fe-08d2276b2a76" />

# Feature Encoding
```
binary_map_bin1 = {'F':0, 'M':1, 'T':2}  
binary_map_bin2 = {'N':0, 'Y':1}
merged_df['bin_1'] = merged_df['bin_1'].map(binary_map_bin1)
merged_df['bin_2'] = merged_df['bin_2'].map(binary_map_bin2)
```
<img width="760" height="146" alt="image" src="https://github.com/user-attachments/assets/5d51e12d-94e3-4ceb-8ae1-4bacd8aaf407" />

# Ordinal Encoding
```
ord2_mapping = {'Cold':0, 'Warm':1, 'Hot':2, 'Very Hot':3}
education_mapping = {'High School':0, 'Diploma':1, 'Bachelors':2, 'Masters':3, 'PhD':4}
merged_df['ord_2'] = merged_df['ord_2'].map(ord2_mapping)
merged_df['Ord_1'] = merged_df['Ord_1'].map(ord2_mapping)
merged_df['Ord_2'] = merged_df['Ord_2'].map(education_mapping)
```
<img width="976" height="140" alt="image" src="https://github.com/user-attachments/assets/baed78e7-217d-4187-bef3-b77ead18c6b9" />

# Feature Transformation

# Positive skew → log:
```
for col in ['Moderate Positive Skew','Highly Positive Skew']:
    merged_df[col+'_log'] = np.log1p(merged_df[col])
```
<img width="773" height="79" alt="image" src="https://github.com/user-attachments/assets/5b611ce2-1b28-4853-894c-612d3b39cc6f" />

# Negative skew → square:
```
for col in ['Moderate Negative Skew','Highly Negative Skew']:
    merged_df[col+'_sq'] = np.square(merged_df[col])
```
<img width="766" height="99" alt="image" src="https://github.com/user-attachments/assets/9026bc39-8868-4773-b7b2-2f9723569062" />

# Scale transformed features
```
scaler = StandardScaler()
scaled_cols = ['Moderate Positive Skew_log','Highly Positive Skew_log',
               'Moderate Negative Skew_sq','Highly Negative Skew_sq']
merged_df[scaled_cols] = scaler.fit_transform(merged_df[scaled_cols])
```
<img width="859" height="134" alt="image" src="https://github.com/user-attachments/assets/ee0d9eef-608e-4b03-86cd-ad652e9a3bd6" />

# Visualization of transformations
```
def plot_before_after(original, transformed, title):
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    sns.histplot(original, kde=True, bins=30, color="skyblue")
    plt.title(f"Before Transformation: {title}")
    
    plt.subplot(1,2,2)
    sns.histplot(transformed, kde=True, bins=30, color="salmon")
    plt.title(f"After Transformation: {title}")
    
    plt.tight_layout()
    plt.show()
```
<img width="846" height="346" alt="image" src="https://github.com/user-attachments/assets/33e46699-8520-4f50-bb1b-5521dcd5fcb7" />

# Plot for each skewed feature
```
plot_before_after(merged_df['Moderate Positive Skew'], merged_df['Moderate Positive Skew_log'], "Moderate Positive Skew")
plot_before_after(merged_df['Highly Positive Skew'], merged_df['Highly Positive Skew_log'], "Highly Positive Skew")
plot_before_after(merged_df['Moderate Negative Skew'], merged_df['Moderate Negative Skew_sq'], "Moderate Negative Skew")
plot_before_after(merged_df['Highly Negative Skew'], merged_df['Highly Negative Skew_sq'], "Highly Negative Skew")
```
<img width="1317" height="497" alt="image" src="https://github.com/user-attachments/assets/a073b9f6-6d33-47b3-8f06-0f065858f2d4" />

<img width="1292" height="498" alt="image" src="https://github.com/user-attachments/assets/48a48814-cc75-4ac7-8fdc-1e86e7622490" />

<img width="1278" height="498" alt="image" src="https://github.com/user-attachments/assets/7d9334d9-c946-47c9-ad47-cfe736aba1be" />

<img width="1291" height="484" alt="image" src="https://github.com/user-attachments/assets/ed38553e-7bfe-4379-b13f-fbcb68a26905" />

# Save Final Dataset
```
merged_df.to_csv("Processed_Dataset.csv", index=False)

print("✅ Processing complete! Transformed dataset saved as 'Processed_Dataset.csv'")
```
<img width="977" height="153" alt="image" src="https://github.com/user-attachments/assets/08842d48-7925-41da-86d3-e5675c3b7b31" />



# RESULT:

To read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully executed

# SUMMARY:

In this experiment, we worked with three different datasets that contained a mixture of categorical and numerical features. The goal was to prepare the data for further analysis and insights in data science.
We transformed raw, inconsistent data into a well-structured dataset that can now be used for exploratory data analysis (EDA), descriptive statistics, visualization in data science.   

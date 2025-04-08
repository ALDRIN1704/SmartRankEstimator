import pandas as pd

# Load raw CSV files
df_2022 = pd.read_csv("NIRF-2022.csv")
df_2023 = pd.read_csv("Overall2023.csv")
df_2024 = pd.read_csv("NIRF-2024.csv")

# Function to clean and standardize the 2022 and 2024 datasets
def clean_nirf_dataset(df, year):
    df = df.drop(index=0).reset_index(drop=True)
    df_cleaned = pd.DataFrame()

    # Use correct column names from the raw file
    df_cleaned['Institute ID'] = df['INSTITUTE  ID']
    df_cleaned['Institute Name'] = df['NAME OF THE INSTITUTE']
    df_cleaned['City'] = df['CITY']
    df_cleaned['TLR(100)'] = pd.to_numeric(df['SCORE'], errors='coerce')
    df_cleaned['RPC(100)'] = pd.to_numeric(df['TOTAL/100'], errors='coerce')
    df_cleaned['GO(100)'] = pd.to_numeric(df['TOTAL/100.2'], errors='coerce')
    df_cleaned['OI(100)'] = pd.to_numeric(df['OI/100'], errors='coerce')
    df_cleaned['Perception(100)'] = pd.to_numeric(df['PR/100'], errors='coerce')
    df_cleaned['Score'] = pd.to_numeric(df['TOTAL/100.3'], errors='coerce')
    df_cleaned['Ranking'] = pd.to_numeric(df['RANK'], errors='coerce')
    df_cleaned['Year'] = year

    return df_cleaned

# Clean 2022 and 2024 datasets
df_2022_clean = clean_nirf_dataset(df_2022, 2022)
df_2024_clean = clean_nirf_dataset(df_2024, 2024)

# Prepare and clean 2023 dataset
df_2023_clean = df_2023.copy()
df_2023_clean['Year'] = 2023
df_2023_clean = df_2023_clean[[
    'Institute ID', 'Institute Name', 'City',
    'TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)',
    'Score', 'Ranking', 'Year'
]]

# Merge all datasets
nirf_master_df = pd.concat([df_2022_clean, df_2023_clean, df_2024_clean], ignore_index=True)

# Save merged cleaned dataset
nirf_master_df.to_csv("Cleaned_NIRF_2022_2023_2024.csv", index=False)
print("✅ Cleaned dataset saved as 'Cleaned_NIRF_2022_2023_2024.csv'")

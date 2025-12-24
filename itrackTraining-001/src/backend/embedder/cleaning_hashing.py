import pandas as pd
import os
from dotenv import load_dotenv
import hashlib
import warnings
from datetime import datetime # datetime is not directly used in this logic, but good to keep if needed later
from pandas.errors import SettingWithCopyWarning

# Suppress pandas warning about column assignment (optional, but cleaner output)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# ===================================================================
# CORE HASHING AND PROCESSING LOGIC (Your Provided Code)
# ===================================================================

def md5_hash(text):
    """Generates an MD5 hash for a given text string."""
    if pd.isna(text):
        text = ""
    return hashlib.md5(str(text).strip().encode('utf-8')).hexdigest()


def get_processed_dataframes(main_folder_path=None,
                             trainee_output='Trainee_Processed.csv',
                             trainer_output='Trainer_Processed.csv'):
    """
    Processes the raw Excel files -> cleans -> hashes -> returns (trainee_df, trainer_df).
    """
    
    # Load .env if not loaded
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
    if main_folder_path is None:
        main_folder_path = os.getenv("MAIN_FOLDER")
    # Define folders
    trainees_folder = os.path.join(main_folder_path, "Trainees")
    trainer_folder = os.path.join(main_folder_path, "Trainer")
    output_folder = os.path.join(main_folder_path, "tests")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(trainees_folder) and not os.path.exists(trainer_folder):
        print("Error: Neither 'Trainees' nor 'Trainer' folder found!")
        return pd.DataFrame(), pd.DataFrame()

    print("Starting data processing...\n")

    # ===================================================================
    # 1. TRAINEE DATA
    # ===================================================================
    trainee_df = pd.DataFrame()
    if os.path.exists(trainees_folder):
        trainee_files = [f for f in os.listdir(trainees_folder) if f.lower().endswith('.xlsx')]
        if trainee_files:
            print("1. Processing Trainee files...")
            dfs = []
            for file_name in trainee_files:
                path = os.path.join(trainees_folder, file_name)
                try:
                    df = pd.read_excel(path)
                    if df.empty:
                        print(f"    [Skipped] {file_name} -> empty")
                        continue

                    trainee_name = os.path.splitext(file_name)[0].strip()
                    df = df.copy()
                    df['Trainee_Name'] = trainee_name
                    dfs.append(df)
                    print(f"    Loaded: {file_name} -> Trainee_Name = '{trainee_name}' | Shape: {df.shape}")
                except Exception as e:
                    print(f"    Failed: {file_name} -> {e}")

            if dfs:
                trainee_df = pd.concat(dfs, ignore_index=True, sort=False)
                print(f"    Combined: {trainee_df.shape}")

                print("    Filling blanks -> numbers: 0 | text: 'N/A'...")
                for col in trainee_df.columns:
                    if pd.api.types.is_numeric_dtype(trainee_df[col]):
                        trainee_df[col] = trainee_df[col].fillna(0)
                    else:
                        trainee_df[col] = trainee_df[col].fillna('N/A')

                print("    Cleaning text...")
                def clean_text(s):
                    return (s.astype(str)
                            .str.strip()
                            .str.replace(r'[\t\n\r]+', ' ', regex=True)
                            .str.replace(r'\s{2,}', ' ', regex=True)
                            .replace('nan', 'N/A'))
                
                obj_cols = trainee_df.select_dtypes(include=['object']).columns
                trainee_df.loc[:, obj_cols] = trainee_df[obj_cols].apply(clean_text)

                before = len(trainee_df)
                trainee_df.drop_duplicates(inplace=True)
                print(f"    Removed {before - len(trainee_df)} duplicate rows.\n")

                identity = ['Trainee_Name', 'Employee_id', 'Email_address', 'Last_log_date', 'Training_id']
                for col in identity:
                    if col not in trainee_df.columns:
                        trainee_df[col] = 'N/A'

                trainee_df['data_hash'] = trainee_df[identity].astype(str).agg('|||'.join, axis=1).apply(md5_hash)
                exclude = ['data_hash', 'version_hash']
                version_cols = [c for c in trainee_df.columns if c not in exclude]
                trainee_df['version_hash'] = trainee_df[version_cols].astype(str).agg('|||'.join, axis=1).apply(md5_hash)

                hash_cols = ['data_hash', 'version_hash']
                other_cols = [c for c in trainee_df.columns if c not in hash_cols]
                trainee_df = trainee_df[other_cols + hash_cols]

        else:
            print("No Trainee files found -> skipping\n")
    else:
        print("'Trainees' folder missing -> skipping\n")

    # ===================================================================
    # 2. TRAINER DATA
    # ===================================================================
    trainer_df = pd.DataFrame()
    if os.path.exists(trainer_folder):
        trainer_files = [f for f in os.listdir(trainer_folder) if f.lower().endswith('.xlsx')]
        if trainer_files:
            print("2. Processing Trainer master file...")
            sheets = []
            for file_name in trainer_files:
                path = os.path.join(trainer_folder, file_name)
                try:
                    print(f"    Reading -> {file_name}")
                    xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
                    for sheet_name, df in xls.items():
                        if df.empty:
                            continue
                        df = df.copy()
                        df['Trainee_Name'] = sheet_name.strip()
                        sheets.append(df)
                        print(f"    Sheet: '{sheet_name}' -> Shape: {df.shape}")
                except Exception as e:
                    print(f"    Failed: {file_name} -> {e}")

            if sheets:
                trainer_df = pd.concat(sheets, ignore_index=True, sort=False)
                print(f"    Combined: {trainer_df.shape}")

                print("    Filling blanks -> numbers: 0 | text: 'N/A'...")
                for col in trainer_df.columns:
                    if pd.api.types.is_numeric_dtype(trainer_df[col]):
                        trainer_df[col] = trainer_df[col].fillna(0)
                    else:
                        trainer_df[col] = trainer_df[col].fillna('N/A')

                print("    Cleaning text...")
                def clean_text(s):
                    return (s.astype(str)
                            .str.strip()
                            .str.replace(r'[\t\n\r]+', ' ', regex=True)
                            .str.replace(r'\s{2,}', ' ', regex=True)
                            .replace('nan', 'N/A'))
                
                obj_cols = trainer_df.select_dtypes(include=['object']).columns
                trainer_df.loc[:, obj_cols] = trainer_df[obj_cols].apply(clean_text)

                before = len(trainer_df)
                trainer_df.drop_duplicates(inplace=True)
                print(f"    Removed {before - len(trainer_df)} duplicate rows.\n")
                
                # Check for 'Employee_id' before using it for hashing (standardizing)
                if 'Employee_id' not in trainer_df.columns:
                    trainer_df['Employee_id'] = 'N/A'

                trainer_df['data_hash'] = (
                    trainer_df['Trainee_Name'].astype(str) + '|||' + 
                    trainer_df['Employee_id'].astype(str)
                ).apply(md5_hash)

                exclude = ['data_hash', 'version_hash']
                version_cols = [c for c in trainer_df.columns if c not in exclude]
                trainer_df['version_hash'] = trainer_df[version_cols].astype(str).agg('|||'.join, axis=1).apply(md5_hash)

        else:
            print("No Trainer file -> skipping")
    else:
        print("'Trainer' folder missing -> skipping")

    print("\n" + "="*90)
    print("Processing complete! Returning cleaned DataFrames.")
    print("="*90)

    # Return both DataFrames for external use
    return trainee_df, trainer_df




# ===================================================================
# GENERIC FUNCTION TO RETURN CLEANED DATAFRAMES
# ===================================================================

def get_cleaned_dataframes():
    """
    Returns cleaned trainee and trainer DataFrames directly from memory.
    """
    trainee_df, trainer_df = get_processed_dataframes()
    return trainee_df, trainer_df
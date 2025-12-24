# cleaning_hashing.py
# Loads, cleans, deduplicates, and hashes Trainee & Trainer Excel data
# Returns two processed pandas DataFrames ready for indexing

import pandas as pd
import os
import hashlib
import re
import nltk
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources globally for efficiency
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def md5_hash(text):
    """Helper to create MD5 hash; handles NaN safely"""
    if pd.isna(text):
        text = ""
    return hashlib.md5(str(text).strip().encode('utf-8')).hexdigest()


def normalize_text(text):
    """
    Advanced text normalization for descriptive fields:
    """
    if pd.isna(text) or str(text).strip() in ['', 'N/A', 'nan']:
        return 'N/A'

    text = str(text).lower().strip()

    try:
        tokens = word_tokenize(text)
        cleaned_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
        ]
        return ' '.join(cleaned_tokens) if cleaned_tokens else 'N/A'
    except Exception as e:
        print(f"Warning: normalize_text failed: {e}")
        return str(text).strip()  # fallback


def _post_process(df, identity_columns):
    """Shared post-processing: cleaning, hashing, deduplication, page_content creation"""
    if df.empty:
        return df

    print(f"   Combined raw records: {df.shape[0]} rows")

    # 1. Fill missing values
    print("   Filling missing values → numeric: 0 | text: 'N/A'")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('N/A')

    # 2. Clean text + Advanced normalization on specific columns
    print("   Cleaning text columns...")
    obj_cols = df.select_dtypes(include='object').columns
    def basic_clean(s):
        return (
            s.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[\r\n\t]+', ' ', regex=True)
            .str.replace(r'\s{2,}', ' ', regex=True)
            .replace('nan', 'N/A')
            .replace('NAN', 'N/A')
            .replace('n/a', 'N/A')
            .replace('null', 'N/A')
        )
    df[obj_cols] = df[obj_cols].apply(basic_clean)
    print("after cleaning process: ")
    print(df)

    # Advanced normalization on descriptive columns
    descriptive_columns = []
    trainee_cols = ['PTO_Reason', 'Org_Participation', 'Certifications', 'Additional_Skills']
    trainer_cols = ['To_Focus', 'Trainer_Comments']
    descriptive_columns = [col for col in trainee_cols + trainer_cols if col in df.columns]
    
    if descriptive_columns:
        print(f"   Applying advanced normalization to: {descriptive_columns}")
        for col in descriptive_columns:
            df[col] = df[col].apply(normalize_text)

    # 3. Standardize date columns
    print("   Standardizing date columns...")
    date_keywords = ['date', 'log', 'start', 'expected', 'end', 'review', 'reporting']
    date_candidates = [col for col in df.columns if any(k in col.lower() for k in date_keywords)]
    for col in date_candidates:
        converted = pd.to_datetime(df[col], errors='coerce')
        invalid_count = converted.isna().sum()
        if invalid_count > 0:
            print(f"      {col}: {invalid_count} invalid dates → set to 'N/A'")
        df[col] = converted.dt.strftime('%Y-%m-%d').fillna('N/A')

    # 4. Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   Removed {before - len(df)} exact duplicate rows")

    # 5. Ensure identity columns exist
    for col in identity_columns:
        if col not in df.columns:
            df[col] = 'N/A'

    # 6. Create data_hash (unique identifier based on identity)
    print(f"   Creating data_hash using: {identity_columns}")
    df['data_hash'] = (
        df[identity_columns]
        .astype(str)
        .agg('|||'.join, axis=1)
        .apply(md5_hash)
    )

    # 7. Create version_hash (detects content changes)
    version_cols = [c for c in df.columns if c not in ['data_hash', 'version_hash']]
    df['version_hash'] = (
        df[version_cols]
        .astype(str)
        .agg('|||'.join, axis=1)
        .apply(md5_hash)
    )

    # 8. Create page_content for RAG embedding
    print("   Creating page_content column for document indexing...")
    content_cols = [c for c in df.columns if c not in ['data_hash', 'version_hash', 'page_content']]
    df['page_content'] = df.apply(
        lambda row: ' '.join([
            f"{col}: {row[col]}" for col in content_cols if pd.notna(row[col]) and str(row[col]).strip() != ''
        ]),
        axis=1
    )

    # 9. Reorder columns: page_content first, then others, hashes last
    final_order = ['page_content'] + [c for c in df.columns if c not in ['page_content', 'data_hash', 'version_hash']] + ['data_hash', 'version_hash']
    df = df[final_order]

    print(f"   Final processed records: {len(df)} | Unique data_hash: {df['data_hash'].nunique()}\n")
    return df


def get_processed_dataframes(main_folder_path):
    """
    Main function: Loads and processes Trainee & Trainer Excel files.
    Returns: (trainee_df, trainer_df) – both fully cleaned and hashed.
    """
    trainees_folder = os.path.join(main_folder_path, "Trainees")
    trainer_folder = os.path.join(main_folder_path, "Trainer")

    if not os.path.exists(trainees_folder) and not os.path.exists(trainer_folder):
        print("=" * 90)
        print("   CRITICAL ERROR: Neither 'Trainees' nor 'Trainer' folder exists!")
        print("   Pipeline will now exit.")
        print("=" * 90)
        sys.exit(1)

    print("Starting data processing...\n")

    # ===================================================================
    # 1. PROCESS TRAINEE DATA
    # ===================================================================
    trainee_df = pd.DataFrame()

    # Hardcoded expected columns for trainees (Trainee_Name is added separately)
    TRAINEE_EXPECTED_COLUMNS = [
        'Employee_id', 'Email_address', 'Training_id', 'Last_log_date',
        'PTO', 'PTO_Reason', 'WFH', 'Task_Name', 'Task_Completed',
        'Start_Date', 'Expected_date', 'End_Date',
        'Org_Participation', 'Certifications', 'Additional_Skills'
    ]

    if os.path.exists(trainees_folder):
        trainee_files = [f for f in os.listdir(trainees_folder) if f.lower().endswith('.xlsx')]
        if not trainee_files:
            print("No Trainee files found → skipping\n")
        else:
            print("1. Processing Trainee files...")
            for file_name in trainee_files:
                path = os.path.join(trainees_folder, file_name)
                try:
                    df = pd.read_excel(path)
                    if df.empty:
                        print(f"   [Skipped] {file_name} → empty")
                        continue
                    #Taking Trainee name from the file name
                    trainee_name = os.path.splitext(file_name)[0].strip()
                    print(f"   Loaded: {file_name} → Trainee_Name = '{trainee_name}' | Shape: {df.shape}")

                    # Fixed schema enforcement
                    current_cols = set(df.columns)
                    missing = set(TRAINEE_EXPECTED_COLUMNS) - current_cols
                    # Exclude Trainee_Name from extra check
                    extra = current_cols - set(TRAINEE_EXPECTED_COLUMNS) - {'Trainee_Name'}

                    if missing or extra:
                        print(f"   [Schema Adjustment] {file_name}")
                        if missing:
                            print(f"      Missing columns: {missing} → adding as 'N/A'")
                            for col in missing:
                                df[col] = 'N/A'
                        if extra:
                            print(f"      Extra columns: {extra} → dropping")
                            df = df.drop(columns=extra)

                    # Always set Trainee_Name (overwrite if it existed)
                    df['Trainee_Name'] = trainee_name

                    # Reorder: expected columns first, then Trainee_Name last
                    final_cols = TRAINEE_EXPECTED_COLUMNS + ['Trainee_Name']
                    df = df[final_cols]  # Guaranteed to have all

                    trainee_df = pd.concat([trainee_df, df], ignore_index=True, sort=False)

                except Exception as e:
                    print(f"   Failed to load {file_name}: {e}")

            if len(trainee_df) > 0:
                trainee_identity = ['Trainee_Name', 'Employee_id', 'Email_address', 'Training_id', 'Last_log_date']
                trainee_df = _post_process(trainee_df, identity_columns=trainee_identity)

    else:
        print("'Trainees' folder missing → skipping\n")

    # ===================================================================
    # 2. PROCESS TRAINER DATA
    # ===================================================================
    trainer_df = pd.DataFrame()

    # Hardcoded expected columns for trainers (Trainee_Name added separately)
    TRAINER_EXPECTED_COLUMNS = [
        'Employee_id', 'Training_id', 'Trainer_Name', 'attending_calls',
        'reporting_time', 'Review_date',
        'Technical_Aggregate_Python', 'Technical_Aggregate_Snowflake',
        'Technical_Aggregate_Databricks', 'Problem_Solving',
        'Communication_Score', 'Professionalism_Score', 'Presentation_skills',
        'Team_Collaboration', 'Leadership_Score',
        'To_Focus', 'Trainer_Comments'
    ]

    if os.path.exists(trainer_folder):
        trainer_files = [f for f in os.listdir(trainer_folder) if f.lower().endswith('.xlsx')]
        if not trainer_files:
            print("No Trainer files found → skipping\n")
        else:
            print("2. Processing Trainer files (multi-sheet support)...")
            for file_name in trainer_files:
                path = os.path.join(trainer_folder, file_name)
                try:
                    print(f"   Reading → {file_name}")
                    xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')

                    for sheet_name, df in xls.items():
                        if df.empty:
                            continue

                        trainee_name = sheet_name.strip()
                        print(f"      Sheet: '{trainee_name}' → Shape: {df.shape}")

                        # Fixed schema enforcement
                        current_cols = set(df.columns)
                        missing = set(TRAINER_EXPECTED_COLUMNS) - current_cols
                        # Exclude Trainee_Name from extra check
                        extra = current_cols - set(TRAINER_EXPECTED_COLUMNS) - {'Trainee_Name'}

                        if missing or extra:
                            print(f"      [Schema Adjustment] Sheet '{trainee_name}'")
                            if missing:
                                print(f"          Missing columns: {missing} → adding as 'N/A'")
                                for col in missing:
                                    df[col] = 'N/A'
                            if extra:
                                print(f"          Extra columns: {extra} → dropping")
                                df = df.drop(columns=extra)

                        # Always set Trainee_Name (overwrite if it existed)
                        df['Trainee_Name'] = trainee_name

                        # Reorder: expected columns first, then Trainee_Name last
                        final_cols = TRAINER_EXPECTED_COLUMNS + ['Trainee_Name']
                        df = df[final_cols]  # Guaranteed to have all

                        trainer_df = pd.concat([trainer_df, df], ignore_index=True, sort=False)

                except Exception as e:
                    print(f"   Failed: {file_name} → {e}")

            if len(trainer_df) > 0:
                trainer_identity = ['Trainee_Name', 'Employee_id', 'Training_id']
                trainer_df = _post_process(trainer_df, identity_columns=trainer_identity)

    else:
        print("'Trainer' folder missing → skipping\n")

    print("=" * 90)
    print("DATA PROCESSING COMPLETE!")
    print("=" * 90)

    return trainee_df, trainer_df


# ==================================================================
# CSV SAVE (kept for testing/debugging)
# ==================================================================
def save_to_csv(trainee_df, trainer_df, output_folder=None):
    """
    Saves the processed trainee and trainer DataFrames to CSV files (for testing).
    """
    if output_folder is None:
        output_folder = r"C:\Users\devavaishnee.m\Desktop\itrack-chatbot\itrackTraining-001\tests"
    
    os.makedirs(output_folder, exist_ok=True)
    
    trainee_path = os.path.join(output_folder, "processed_trainees.csv")
    trainer_path = os.path.join(output_folder, "processed_trainers.csv")
    
    trainee_df.to_csv(trainee_path, index=False, encoding='utf-8')
    print(f"   Trainee CSV saved: {trainee_path} | Rows: {len(trainee_df)}")
    
    trainer_df.to_csv(trainer_path, index=False, encoding='utf-8')
    print(f"   Trainer CSV saved: {trainer_path} | Rows: {len(trainer_df)}")


# ==================================================================
# Manual testing entry point (kept for standalone runs)
# ==================================================================
def main():
    """For manual testing: run this file directly"""
    main_folder = r"C:\Users\devavaishnee.m\Desktop\itrack-chatbot\itrackTraining-001"

    print("RUNNING CLEANING & HASHING PIPELINE (TEST MODE)\n")
    trainee_df, trainer_df = get_processed_dataframes(main_folder)

    print(f"\nFinal Results:")
    print(f"   Trainee records : {len(trainee_df):,}")
    print(f"   Trainer records : {len(trainer_df):,}")

    print("\nSaving processed data to CSV files for inspection...")
    save_to_csv(trainee_df, trainer_df)

    print("\nReady for indexing → run indexing_chunking.py next!")


if __name__ == "__main__":
    main()
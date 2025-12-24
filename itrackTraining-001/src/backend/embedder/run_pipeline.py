# run_pipeline.py
# Master orchestrator for the iTrack RAG knowledge base update pipeline
# Actual flow: 1. Cleaning & Hashing → 2. Embedding & Smart Upsert (chunking done inside)

import os
import sys
import time
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MAIN_FOLDER = r"C:\Users\devavaishnee.m\Desktop\itrack-chatbot\itrackTraining-001"

# Add project root to path if needed
if MAIN_FOLDER not in sys.path:
    sys.path.insert(0, MAIN_FOLDER)

# Import pipeline functions
from cleaning_hashing import get_processed_dataframes
from embedd_loadTochroma import upsert_to_chroma_and_postgres


def main():
    print("\n" + "=" * 90)
    print("iTRACK RAG KNOWLEDGE BASE PIPELINE".center(90))
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    total_start = time.time()

    # ===================================================================
    # STEP 1: Cleaning & Hashing
    # ===================================================================
    print("\n" + "=" * 90)
    print("STEP 1: Cleaning & Hashing".center(90))
    print("=" * 90)

    start = time.time()

    try:
        trainee_df, trainer_df = get_processed_dataframes(MAIN_FOLDER)
    except SystemExit:
        print("\nPipeline stopped: Required folders (Trainees/Trainer) missing.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError in cleaning & hashing step: {e}")
        sys.exit(1)

    total_records = len(trainee_df) + len(trainer_df)
    print(f"✓ SUCCESS: 1. Cleaning & Hashing ({time.time() - start:.1f}s)")
    print(f"   Records → Trainees: {len(trainee_df):,}, Trainers: {len(trainer_df):,}, Total: {total_records:,}")

    if total_records == 0:
        print("\nNo data found in Excel files. Nothing to process — stopping pipeline.")
        print("=" * 90)
        sys.exit(0)

    # ===================================================================
    # STEP 2: Embedding, Chunking & Smart Upsert (all in one)
    # ===================================================================
    print("\n" + "=" * 90)
    print("STEP 2: Embedding, Chunking & Smart Upsert".center(90))
    print("=" * 90)

    start = time.time()
    upsert_to_chroma_and_postgres(MAIN_FOLDER)  # This handles everything after cleaning
    print(f"✓ SUCCESS: 2. Embedding & Upsert ({time.time() - start:.1f}s)")

    # ===================================================================
    # Final Summary
    # ===================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 90)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!".center(90))
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
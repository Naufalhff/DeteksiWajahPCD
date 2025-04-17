import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Configuration
CROPPED_CSV = 'cropped_dataset.csv'
SPLIT_DIR = 'splits'

def create_folders():
    """Create train/val/test folders"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

def clean_data(df):
    """Clean and standardize data"""
    # Fix the path_gambar column by handling special characters in paths
    df['clean_path'] = df['path_gambar'].apply(lambda x: x.strip('"'))
    
    # Drop rows with invalid clean_path
    df = df.dropna(subset=['clean_path'])
    
    # Standardize values
    df['sudut'] = df['sudut'].str.lower().str.strip()
    df['ekspresi'] = df['ekspresi'].str.replace('marahindoor', 'marah').str.strip()
    df['ekspresi'] = df['ekspresi'].str.replace('kaget', 'terkejut').str.strip()
    
    # Ensure pencahayaan column is clean
    df['pencahayaan'] = df['pencahayaan'].fillna('indoor').str.strip()
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The cleaned DataFrame is empty. Please check the input data.")
    
    return df

def split_data(df):
    """Stratified split by ethnicity and expression"""
    major_ethnics = ['Jawa', 'Sunda']
    major = df[df['suku'].isin(major_ethnics)]
    minor = df[~df['suku'].isin(major_ethnics)]
    
    # Initial split
    train_val, test = train_test_split(
        major,
        test_size=0.15,
        stratify=major[['suku', 'ekspresi']],
        random_state=42
    )
    
    # Secondary split
    train, val = train_test_split(
        train_val,
        test_size=0.176,  # Approximately 15% of the original dataset
        stratify=train_val[['suku', 'ekspresi']],
        random_state=42
    )
    
    # Combine minorities with test set
    return train, val, pd.concat([test, minor])

def copy_files(df, split_name):
    """Organize files into structured folders"""
    success = 0
    failed = 0
    
    for _, row in df.iterrows():
        # Create destination path
        dst_dir = os.path.join(
            SPLIT_DIR, 
            split_name, 
            row['suku'], 
            row['nama']
        )
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get source file path
        src_path = row['clean_path']
        
        # Check if file exists
        if not os.path.exists(src_path):
            # Try with current working directory as base
            full_path = os.path.join(os.getcwd(), src_path)
            if not os.path.exists(full_path):
                print(f"File not found: {src_path}")
                failed += 1
                continue
            else:
                src_path = full_path
                
        # Copy file
        try:
            # Extract filename while preserving commas in the filename
            filename = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            success += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            failed += 1
    
    print(f"{split_name}: Successfully copied {success} files, failed to copy {failed} files")

def main():
    print("Starting dataset splitting process...")
    
    # Check if CSV exists
    if not os.path.exists(CROPPED_CSV):
        print(f"Error: CSV file '{CROPPED_CSV}' not found!")
        return
        
    # Load and prepare data
    print(f"Loading data from {CROPPED_CSV}...")
    df = pd.read_csv(CROPPED_CSV)
    print(f"Loaded {len(df)} entries")
    
    # Clean data
    print("Cleaning and standardizing data...")
    df = clean_data(df)
    print(f"After cleaning: {len(df)} valid entries")
    
    # Create folder structure
    print("Creating folder structure...")
    create_folders()
    
    # Split dataset
    print("Splitting dataset...")
    train, val, test = split_data(df)
    
    # Distribute files
    print("\nCopying files to respective folders:")
    for split, data in [('train', train), ('val', val), ('test', test)]:
        print(f"\nProcessing {split} set ({len(data)} images)...")
        copy_files(data, split)
        # Save metadata without the temporary clean_path column
        metadata_file = os.path.join(SPLIT_DIR, f'{split}_metadata.csv')
        data.drop(columns=['clean_path']).to_csv(metadata_file, index=False)
        print(f"Saved metadata to {metadata_file}")
    
    # Print summary
    print("\n=== Dataset Split Summary ===")
    print(f"Train: {len(train)} images")
    print(f"Validation: {len(val)} images")
    print(f"Test: {len(test)} images (including minority ethnics)")
    
    # Show ethnic distribution
    print("\nEthnic Distribution:")
    for name, df in [('Train', train), ('Validation', val), ('Test', test)]:
        print(f"\n{name}:")
        print(df['suku'].value_counts())

if __name__ == '__main__':
    main()
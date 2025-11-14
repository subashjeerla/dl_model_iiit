# improved_data_processing.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_clean_data():
    """Load and clean all three datasets with consistent formatting"""
    
    # Load datasets
    ap_df = pd.read_csv("/home/me3-lab/Project/data/AP_EAMCET_Cleaned_Merged.csv")
    ts_df = pd.read_csv("/home/me3-lab/Project/data/TSEAMCET_2021_2022_2023_merged_clean.csv")
    jee_df = pd.read_csv("/home/me3-lab/Project/data/JEE_data.csv")
    
    # Standardize AP data
    ap_records = []
    ap_cat_cols = [c for c in ap_df.columns if any(k in c.upper() for k in ['BOYS','GIRLS','OC','BC','SC','ST','EWS'])]
    
    for _, row in ap_df.iterrows():
        for cat_col in ap_cat_cols:
            try:
                cutoff = float(row[cat_col])
                if cutoff > 0:  # Valid cutoff
                    ap_records.append({
                        'exam': 'AP',
                        'college': row.get('NAME_OF_THE_INSTITUTION', 'Unknown'),
                        'branch': row.get('branch_code', row.get('branch', 'Unknown')),
                        'category': cat_col.upper().strip(),
                        'cutoff': cutoff
                    })
            except (ValueError, TypeError):
                continue
    
    # Standardize TS data
    ts_records = []
    ts_cat_cols = [c for c in ts_df.columns if any(k in c.lower() for k in ['boys','girls','oc','bc','sc','st','ews'])]
    
    for _, row in ts_df.iterrows():
        for cat_col in ts_cat_cols:
            try:
                cutoff = float(row[cat_col])
                if cutoff > 0:
                    ts_records.append({
                        'exam': 'TS',
                        'college': row.get('institute_name', 'Unknown'),
                        'branch': row.get('branch', 'Unknown'),
                        'category': cat_col.upper().strip(),
                        'cutoff': cutoff
                    })
            except (ValueError, TypeError):
                continue
    
    # Standardize JEE data
    jee_records = []
    if 'closing_rank' in jee_df.columns:
        for _, row in jee_df.iterrows():
            try:
                cutoff = float(row['closing_rank'])
                if cutoff > 0:
                    jee_records.append({
                        'exam': 'JEE',
                        'college': row.get('institute_short', row.get('institute', 'Unknown')),
                        'branch': row.get('program_name', row.get('branch', 'Unknown')),
                        'category': str(row.get('category', 'GENERAL')).upper().strip(),
                        'cutoff': cutoff
                    })
            except (ValueError, TypeError):
                continue
    
    # Combine all records
    all_records = ap_records + ts_records + jee_records
    cutoffs_df = pd.DataFrame(all_records)
    
    # Clean category names
    def standardize_category(cat):
        cat = str(cat).upper().strip()
        # Map variations to standard names
        if 'OC' in cat and 'BOYS' in cat: return 'OC_BOYS'
        if 'OC' in cat and 'GIRLS' in cat: return 'OC_GIRLS'
        if 'SC' in cat and 'BOYS' in cat: return 'SC_BOYS'
        if 'SC' in cat and 'GIRLS' in cat: return 'SC_GIRLS'
        if 'ST' in cat and 'BOYS' in cat: return 'ST_BOYS'
        if 'ST' in cat and 'GIRLS' in cat: return 'ST_GIRLS'
        if 'BC' in cat or 'OBC' in cat:
            if 'BOYS' in cat: return 'BC_BOYS'
            if 'GIRLS' in cat: return 'BC_GIRLS'
            return 'BC'
        if 'EWS' in cat: return 'EWS'
        if 'GENERAL' in cat: return 'GENERAL'
        return cat
    
    cutoffs_df['category'] = cutoffs_df['category'].apply(standardize_category)
    
    return cutoffs_df

def create_training_data(cutoffs_df, samples_per_cutoff=10):
    """Create realistic training data from cutoff information"""
    
    student_rows = []
    np.random.seed(42)
    
    for _, row in cutoffs_df.iterrows():
        cutoff = row['cutoff']
        if pd.isna(cutoff) or cutoff <= 0:
            continue
            
        # Generate realistic rank samples around cutoff
        for i in range(samples_per_cutoff):
            # Create ranks with different probabilities
            if i < 3:  # High chance of admission (ranks well below cutoff)
                rank = max(1, int(cutoff * np.random.uniform(0.3, 0.8)))
                admit = 1
            elif i < 6:  # Borderline cases
                rank = max(1, int(cutoff * np.random.uniform(0.8, 1.2)))
                admit = 1 if rank <= cutoff else 0
            else:  # Low chance (ranks above cutoff)
                rank = max(1, int(cutoff * np.random.uniform(1.2, 2.0)))
                admit = 0
            
            # Add some noise to admission decisions
            if np.random.random() < 0.05:  # 5% noise
                admit = 1 - admit
                
            rank_ratio = rank / cutoff
            
            student_rows.append({
                'exam': row['exam'],
                'college': row['college'],
                'branch': row['branch'],
                'category': row['category'],
                'rank': rank,
                'cutoff': cutoff,
                'rank_ratio': rank_ratio,
                'admit': admit
            })
    
    return pd.DataFrame(student_rows)

# Execute data processing
print("Loading and cleaning data...")
cutoffs_df = load_and_clean_data()
print(f"Created cutoff dataframe with {len(cutoffs_df)} entries")

print("Generating training data...")
students_df = create_training_data(cutoffs_df, samples_per_cutoff=8)
print(f"Created training data with {len(students_df)} samples")

# Save processed data
os.makedirs('outputs', exist_ok=True)
cutoffs_df.to_csv('outputs/cutoffs_cleaned.csv', index=False)
students_df.to_csv('outputs/students_training.csv', index=False)

print("Data processing completed!")
# improved_model_training.py - CORRECTED VERSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os  # Make sure this import is here

def train_improved_model():
    """Train an improved admission prediction model"""
    
    # CREATE OUTPUTS FOLDER FIRST
    os.makedirs('outputs', exist_ok=True)
    print("✅ Created outputs/ folder")
    
    # Load processed data
    try:
        students_df = pd.read_csv('outputs/students_training.csv')
        cutoffs_df = pd.read_csv('outputs/cutoffs_cleaned.csv')
        print(f"✅ Loaded data: {len(students_df)} students, {len(cutoffs_df)} cutoffs")
    except FileNotFoundError:
        print("❌ Data files not found. Please run improved_data_processing.py first!")
        return None, None, None, None, None, None
    
    # Encode categorical features
    le_exam = LabelEncoder()
    le_cat = LabelEncoder() 
    le_branch = LabelEncoder()
    le_college = LabelEncoder()
    
    students_df['exam_enc'] = le_exam.fit_transform(students_df['exam'].astype(str))
    students_df['cat_enc'] = le_cat.fit_transform(students_df['category'].astype(str))
    students_df['branch_enc'] = le_branch.fit_transform(students_df['branch'].astype(str))
    students_df['college_enc'] = le_college.fit_transform(students_df['college'].astype(str))
    
    # Feature engineering
    X = students_df[['rank_ratio', 'exam_enc', 'cat_enc', 'branch_enc', 'college_enc']]
    y = students_df['admit']
    
    print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"📊 Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"   {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_model_name = name
    
    print(f"\n🏆 Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")
    
    # Save model and encoders
    try:
        joblib.dump(best_model, 'outputs/best_admission_model.joblib')
        joblib.dump(le_exam, 'outputs/le_exam.joblib')
        joblib.dump(le_cat, 'outputs/le_cat.joblib') 
        joblib.dump(le_branch, 'outputs/le_branch.joblib')
        joblib.dump(le_college, 'outputs/le_college.joblib')
        joblib.dump(scaler, 'outputs/scaler.joblib')
        
        # Save feature names for reference
        feature_names = ['rank_ratio', 'exam_enc', 'cat_enc', 'branch_enc', 'college_enc']
        joblib.dump(feature_names, 'outputs/feature_names.joblib')
        
        print("✅ All models and encoders saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return None, None, None, None, None, None
    
    return best_model, le_exam, le_cat, le_branch, le_college, scaler

# Training execution with error handling
if __name__ == "__main__":
    print("🚀 Starting model training...")
    model, le_exam, le_cat, le_branch, le_college, scaler = train_improved_model()
    
    if model is not None:
        print("🎉 Model training completed successfully!")
        
        # Verify files were created
        import glob
        output_files = glob.glob('outputs/*.joblib')
        print(f"📁 Created {len(output_files)} joblib files in outputs/ folder")
    else:
        print("💥 Model training failed!")
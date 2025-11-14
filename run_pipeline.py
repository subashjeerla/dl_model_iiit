# run_pipeline.py
import subprocess
import sys
import os

def run_pipeline():
    """Run the complete data processing and training pipeline"""
    
    print("🚀 Starting College Admission Predictor Pipeline...")
    
    try:
        # Step 1: Data Processing
        print("\n📊 Step 1: Processing data...")
        from improved_data_processing import load_and_clean_data, create_training_data
        cutoffs_df = load_and_clean_data()
        students_df = create_training_data(cutoffs_df)
        print("✅ Data processing completed")
        
        # Step 2: Model Training
        print("\n🤖 Step 2: Training model...")
        from improved_model_training import train_improved_model
        train_improved_model()
        print("✅ Model training completed")
        
        # Step 3: Start Flask App
        print("\n🌐 Step 3: Starting web application...")
        print("Starting Flask server on http://localhost:5000")
        os.system("python corrected_app.py")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
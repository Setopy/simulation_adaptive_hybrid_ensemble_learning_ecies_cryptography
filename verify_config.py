import os
from config import CONFIG
import pandas as pd

def verify_configuration():
    print("Verifying project configuration...")
    
    # 1. Check directories
    print("\nChecking directories:")
    for name, path in CONFIG['PATHS'].items():
        exists = os.path.exists(path)
        print(f"{name}: {path} - {'✓ Exists' if exists else '✗ Missing'}")
    
    # 2. Check data files
    print("\nChecking dataset files:")
    try:
        train_file = CONFIG['PATHS']['TRAIN_FILE']
        test_file = CONFIG['PATHS']['TEST_FILE']
        
        print(f"Training file: {train_file}")
        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file)
            print(f"✓ Training set loaded successfully: {df_train.shape} samples")
        else:
            print("✗ Training file not found!")
        
        print(f"\nTesting file: {test_file}")
        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file)
            print(f"✓ Testing set loaded successfully: {df_test.shape} samples")
        else:
            print("✗ Testing file not found!")
            
    except Exception as e:
        print(f"Error accessing data files: {str(e)}")
    
    # 3. Check model configurations
    print("\nChecking model configurations:")
    for model_name, params in CONFIG['MODEL_PARAMS'].items():
        print(f"{model_name}: {len(params)} parameters configured")
    
    # 4. Print training settings
    print("\nTraining settings:")
    print(f"Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"Number of epochs: {CONFIG['NUM_EPOCHS']}")
    print(f"Learning rate: {CONFIG['LEARNING_RATE']}")
    print(f"Early stopping patience: {CONFIG['EARLY_STOPPING_PATIENCE']}")

if __name__ == "__main__":
    verify_configuration()

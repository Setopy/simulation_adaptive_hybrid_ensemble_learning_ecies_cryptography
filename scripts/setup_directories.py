import os
from pathlib import Path

def create_project_structure():
    # Get the current directory where the script is running
    BASE_DIR = Path(__file__).resolve().parent
    
    # Define all directories to create
    directories = [
        'data',
        'models',
        'utils',
        'trainers',
        'results',
        'results/models',
        'results/metrics',
        'results/logs'
    ]
    
    # Create each directory
    for dir_name in directories:
        dir_path = os.path.join(BASE_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create necessary __init__.py files
    init_locations = ['models', 'utils', 'trainers']
    for loc in init_locations:
        init_file = os.path.join(BASE_DIR, loc, '__init__.py')
        with open(init_file, 'w') as f:
            pass  # Create empty __init__.py file
        print(f"Created __init__.py in: {loc}")

if __name__ == "__main__":
    create_project_structure()

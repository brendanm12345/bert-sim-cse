import os
import pandas as pd
import random

def shorten_csv_files(data_folder_path):
    print("Starting to shorten CSV files...")
    shortened_data_folder_path = os.path.join(data_folder_path, "shortened_data")
    if not os.path.exists(shortened_data_folder_path):
        print(f"Creating folder: {shortened_data_folder_path}")
        os.makedirs(shortened_data_folder_path)
    else:
        print(f"Folder already exists: {shortened_data_folder_path}")
    
    csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {data_folder_path}.")
    
    for file_name in csv_files:
        try:
            file_path = os.path.join(data_folder_path, file_name)
            shortened_file_path = os.path.join(shortened_data_folder_path, file_name)
            
            # Sample rows without loading the entire file
            def sample_csv_rows(file_path, sample_size):
                with open(file_path, 'r') as file:
                    total_rows = sum(1 for row in file)
                
                # Adjust for header if your CSV has one
                total_rows -= 1
                
                # Generate random sample of row indices to select
                skip_rows = sorted(random.sample(range(1, total_rows+1), total_rows - sample_size))
                
                # Read the sampled rows
                df_sample = pd.read_csv(file_path, skiprows=skip_rows)
                
                return df_sample
            
            # Use sample_csv_rows to sample 1000 rows from the file
            shortened_df = sample_csv_rows(file_path, 1000)
            shortened_df.to_csv(shortened_file_path, index=False)
            print(f"Processed and shortened {file_name}")
        except Exception as e:
            print(f"Failed to process {file_name}. Error type: {type(e).__name__}, Error message: {e}")

data_folder = "data"
shorten_csv_files(data_folder)

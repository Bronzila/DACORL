from pathlib import Path
import pandas as pd
import argparse

def calculate_accumulated_change(csv_path: Path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort the dataframe by 'run' and 'batch' to ensure correct order
    df = df.sort_values(by=['run', 'batch'])
    
    # Calculate the change in validation loss per batch
    df['valid_loss_change'] = df.groupby('run')['valid_loss'].diff().fillna(0)
    
    # Accumulate the change in validation loss over one run
    df['accumulated_valid_loss_change'] = df.groupby('run')['valid_loss_change'].cumsum()
    
    return df

def print_accumulated_change_per_run(df):
    # Group by 'run' and calculate the final accumulated change for each run
    accumulated_change_per_run = df.groupby('run')['accumulated_valid_loss_change'].last()
    
    for run, accumulated_change in accumulated_change_per_run.items():
        print(f"Run {run}: Accumulated Validation Loss Change = {accumulated_change}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accumulated change in validation loss per batch.')
    parser.add_argument('csv_path', type=Path, help='Path to the CSV file')
    
    args = parser.parse_args()
    
    result_df = calculate_accumulated_change(args.csv_path)
    
    # Print the accumulated change after each run
    print_accumulated_change_per_run(result_df)

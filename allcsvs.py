import pandas as pd
import glob
import os

def combine_csv_files(folder_path, output_file='combined_dataset_.csv'):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f'Found {len(csv_files)} CSV files.')

    # Read and combine all CSV files into a single DataFrame
    combined_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file in the same folder
    output_path = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_path, index=False)
    print(f'Combined CSV saved to {output_path}')

if __name__ == '__main__':
    folder_path = r"C:\Users\Meyssa\OneDrive - University of Ottawa\Desktop\MIA5100 W00 Fndn & App. Machine Learning\Flight_price_prediction\Flight_price_prediction\Best booking date\Datasets"
    combine_csv_files(folder_path)

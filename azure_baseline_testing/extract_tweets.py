import pandas as pd

def keep_first_column(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Keep only the first column
    df_first_column = df.iloc[:, [0]]
    
    # Save the result to a new CSV file
    df_first_column.to_csv(output_csv, index=False)


input_csv = 'data/merged_file.csv'
output_csv = 'data/tweets.csv'
keep_first_column(input_csv, output_csv)

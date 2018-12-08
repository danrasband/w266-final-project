import pandas as pd
import os
from glob import glob

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '{}/output-data'.format(DIR_PATH)
CSV_GLOB = '{}/*.csv'.format(OUTPUT_DIR)

def csv_filepaths():
    return glob(CSV_GLOB)

def df_from_csv(filepath):
    return pd.read_csv(filepath, header=0).set_index('entity_id')

def csv_files_as_dataframe():
    return pd.concat([df_from_csv(filepath) for filepath in csv_filepaths()])

def stitch_csv_files():
    df = csv_files_as_dataframe()
    with open('Y_pred.20181207.csv', 'w') as file:
        df.to_csv(file)
    return

if __name__ == "__main__":
    stitch_csv_files()

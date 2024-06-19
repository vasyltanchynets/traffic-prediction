from sqlalchemy import create_engine
import pandas as pd
import os

# ######################### ELT Pipeline #########################
# Get user and password from environment variable
uid = os.environ.get("PGUID")
pwd = os.environ.get("PGPASS")
server = "localhost"
database = "traffic_prediction"
port = "5432"
dir = r"C:\Users\Dell\PycharmProjects\traffic-prediction\dataset"

# Extract data from .csv file
def extract():
    try:
        # Starting directory
        directory = dir
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            # Get filename without extension
            file_wo_ext = os.path.splitext(filename)[0]
            # Only process .csv files
            if filename.endswith(".csv"):
                f = os.path.join(directory, filename)
                # Checking if it is a file
                if os.path.isfile(f):
                    df = pd.read_csv(f, encoding="windows_1258")
                    # Call to load
                    load(df, file_wo_ext)
    except Exception as e:
        print("Data extract error: " + str(e))

# Load data to postgres
def load(df, tbl):
    try:
        rows_imported = 0
        engine = create_engine(f'postgresql://{uid}:{pwd}@{server}:{port}/{database}')
        print(f'Importing rows {rows_imported} to {rows_imported + len(df)}... ')
        # Save df to postgres
        df.to_sql(f"{tbl}", engine, if_exists='replace', index=False)
        rows_imported += len(df)
        # Add elapsed time to final print out
        print("Data imported successful")
    except Exception as e:
        print("Data load error: " + str(e))

try:
    # Call extract function
    df = extract()
except Exception as e:
    print("Error while extracting data: " + str(e))

import pandas as pd
from src.ingest_data import DataIngestor


class CSVDataIngestor(DataIngestor):
    """Concrete class for CSV file ingestion."""
    
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Read CSV file and return as pandas DataFrame."""
        if not file_path.endswith('.csv'):
            raise ValueError("The provided file is not a .csv file.")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded CSV file: {file_path}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")


class ExcelDataIngestor(DataIngestor):
    """Concrete class for Excel file ingestion."""
    
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Read Excel file and return as pandas DataFrame."""
        if not (file_path.endswith('.xlsx') or file_path.endswith('.xls')):
            raise ValueError("The provided file is not an Excel file.")
        
        try:
            df = pd.read_excel(file_path)
            print(f"Successfully loaded Excel file: {file_path}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

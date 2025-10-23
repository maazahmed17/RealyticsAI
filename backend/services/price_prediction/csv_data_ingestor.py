import pandas as pd
from ingest_data import DataIngestor


class CSVDataIngestor(DataIngestor):
    """Concrete class for CSV file ingestion."""
    
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Read CSV file and return as pandas DataFrame."""
        if not file_path.endswith('.csv'):
            raise ValueError("The provided file is not a .csv file.")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded CSV file: {file_path}")
            print(f"Original data shape: {df.shape}")
            print(f"Original columns: {list(df.columns)}")
            
            # Drop PropertyID if it exists (it's just an identifier, not a feature)
            if 'PropertyID' in df.columns:
                df = df.drop(columns=['PropertyID'])
                print(f"Dropped PropertyID column")
            
            # Standardize column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Map column names to expected format if needed
            column_mapping = {
                'pricelakhs': 'price',
                'totalsqft': 'total_sqft',
                'propertyageyears': 'property_age',
                'floornumber': 'floor_number',
                'totalfloors': 'total_floors',
                'furnishingstatus': 'furnishing_status'
            }
            df = df.rename(columns=column_mapping)
            
            print(f"Final data shape: {df.shape}")
            print(f"Final columns: {list(df.columns)}")
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

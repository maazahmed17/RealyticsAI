import os
import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from various file formats (ZIP, CSV, Excel) using the appropriate DataIngestor."""
    # Auto-detect file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    print(f"Detected file extension: {file_extension}")
    print(f"Processing file: {file_path}")

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    
    print(f"Data loaded successfully with shape: {df.shape}")
    return df
    
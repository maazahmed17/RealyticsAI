"""
Property Data Loader for RealyticsAI
====================================
Loads and processes property data for the recommendation system.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PropertyDataLoader:
    """Loads and processes property data for recommendations"""
    
    NUMERIC_COLS = ["total_sqft", "balcony", "bath", "bathroom", "price"]
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the data loader
        
        Args:
            data_path: Path to the property CSV file
        """
        if data_path is None:
            # Use the copied data file
            base_dir = Path(__file__).parent
            data_path = base_dir / "data" / "dataproperties.csv"
        
        self.data_path = Path(data_path)
        self.df = None
        
    def load_properties(self) -> pd.DataFrame:
        """Load and clean the properties CSV file
        
        Returns:
            DataFrame with cleaned property data
        """
        try:
            logger.info(f"Loading property data from {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Property data file not found: {self.data_path}")
                
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.strip()
            
            logger.info(f"Loaded {len(df)} properties with columns: {list(df.columns)}")
            
            # Map price column (handle trailing space)
            price_col = None
            for col in df.columns:
                if 'price' in col.lower():
                    price_col = col
                    break
            
            if price_col and price_col != 'price':
                df = df.rename(columns={price_col: 'price'})
            
            # Convert numeric columns
            for col in self.NUMERIC_COLS:
                if col in df.columns:
                    if col == "price":
                        # Handle Indian currency format: ₹39,00,000.00 or ?39,00,000.00
                        df[col] = df[col].astype(str).str.replace("₹", "").str.replace("?", "").str.replace(",", "").str.replace(".00", "")
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Create search text for TF-IDF
            search_terms = []
            if "location" in df.columns:
                search_terms.append(df["location"].fillna("").astype(str))
            if "size" in df.columns:
                search_terms.append(df["size"].fillna("").astype(str))
            if "area_type" in df.columns:
                search_terms.append(df["area_type"].fillna("").astype(str))

            if search_terms:
                df["search_text"] = search_terms[0]
                for i in range(1, len(search_terms)):
                    df["search_text"] = df["search_text"] + " " + search_terms[i]
            else:
                df["search_text"] = ""
            
            # Clean and filter data
            df = df.dropna(subset=["price"]).reset_index(drop=True)
            
            # Add PropertyID if not present
            if "PropertyID" not in df.columns:
                df["PropertyID"] = "PROP_" + df.index.astype(str).str.zfill(6)
            
            self.df = df
            logger.info(f"Successfully processed {len(df)} properties")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading property data: {e}")
            raise
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific property by ID
        
        Args:
            property_id: The property ID to lookup
            
        Returns:
            Property data as dictionary or None if not found
        """
        if self.df is None:
            self.load_properties()
            
        property_data = self.df[self.df["PropertyID"] == property_id]
        
        if len(property_data) == 0:
            return None
            
        return property_data.iloc[0].to_dict()
    
    def get_properties_by_location(self, location: str, limit: int = 10) -> pd.DataFrame:
        """Get properties in a specific location
        
        Args:
            location: Location name to search for
            limit: Maximum number of properties to return
            
        Returns:
            DataFrame with matching properties
        """
        if self.df is None:
            self.load_properties()
            
        mask = self.df["location"].str.contains(location, case=False, na=False)
        return self.df[mask].head(limit)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data
        
        Returns:
            Dictionary with data summary
        """
        if self.df is None:
            self.load_properties()
            
        return {
            "total_properties": len(self.df),
            "unique_locations": self.df["location"].nunique(),
            "price_range": {
                "min": float(self.df["price"].min()),
                "max": float(self.df["price"].max()),
                "median": float(self.df["price"].median())
            },
            "size_distribution": self.df["size"].value_counts().head(10).to_dict(),
            "columns": list(self.df.columns)
        }
    
    def filter_properties(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Filter properties based on criteria
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            self.load_properties()
            
        filtered_df = self.df.copy()
        
        if "location" in filters and filters["location"]:
            mask = filtered_df["location"].str.contains(
                filters["location"], case=False, na=False
            )
            filtered_df = filtered_df[mask]
        
        if "bhk" in filters and filters["bhk"]:
            mask = filtered_df["size"].str.contains(
                f"{filters['bhk']} BHK", case=False, na=False
            )
            filtered_df = filtered_df[mask]
        
        if "min_sqft" in filters and filters["min_sqft"]:
            filtered_df = filtered_df[filtered_df["total_sqft"] >= filters["min_sqft"]]
        
        if "max_price" in filters and filters["max_price"]:
            filtered_df = filtered_df[filtered_df["price"] <= filters["max_price"]]
        
        if "min_price" in filters and filters["min_price"]:
            filtered_df = filtered_df[filtered_df["price"] >= filters["min_price"]]
        
        return filtered_df
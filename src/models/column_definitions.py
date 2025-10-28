"""Column definitions and mappings for the real estate dataset"""

COLUMNS = {
    'ID': 'PropertyID',
    'LOCATION': 'Location',
    'BHK': 'BHK',
    'TOTAL_SQFT': 'TotalSqft',
    'BATH': 'Bath',
    'BALCONY': 'Balcony',
    'PRICE': 'Price',
    'AGE': 'PropertyAgeYears',
    'FLOOR_NUM': 'FloorNumber',
    'TOTAL_FLOORS': 'TotalFloors',
    'PARKING': 'Parking',
    'FURNISHING': 'FurnishingStatus',
    'AMENITIES': 'Amenities'
}

NUMERIC_COLUMNS = [
    'BHK', 
    'TotalSqft', 
    'Bath', 
    'Balcony', 
    'Price',
    'PropertyAgeYears',
    'FloorNumber',
    'TotalFloors',
    'Parking'
]

CATEGORICAL_COLUMNS = [
    'Location',
    'FurnishingStatus'
]

TEXT_COLUMNS = [
    'Amenities'
]

TARGET_COLUMN = 'Price'
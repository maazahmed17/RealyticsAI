# backend/services/property_recommendation/recommender.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class RecommendationService:
    def __init__(self):
        """
        Initializes the recommendation service by loading and preparing the property data.
        """
        self.data_path = Path(__file__).parent.parent.parent.parent / "data" / "bengaluru_house_prices.csv"
        self.properties_df = None
        self.feature_matrix = None
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads the dataset and creates a feature matrix for similarity calculations.
        """
        if not self.data_path.exists():
            print(f"Error: Data file not found at {self.data_path}")
            return

        self.properties_df = pd.read_csv(self.data_path)
        # Basic cleaning
        self.properties_df.dropna(subset=['location', 'size', 'bath', 'price'], inplace=True)
        self.properties_df['bhk'] = self.properties_df['size'].str.extract('(\d+)').astype(float)

        # Select features for recommendation
        features = self.properties_df[['bhk', 'bath', 'price']].copy()
        
        # Scale numerical features
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(features)

    def get_recommendations(self, user_preferences: dict, top_n: int = 5) -> list:
        """
        Generates property recommendations based on user preferences.

        Args:
            user_preferences (dict): A dictionary with user's preferences 
                                     (e.g., {'bhk': 2, 'bath': 2, 'budget': 80}).
            top_n (int): The number of recommendations to return.

        Returns:
            list: A list of recommended properties.
        """
        if self.properties_df is None:
            return []

        # Create a user profile vector
        user_vector = pd.DataFrame([user_preferences])[['bhk', 'bath', 'price']]
        user_vector_scaled = StandardScaler().fit(user_vector).transform(user_vector)


        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector_scaled, self.feature_matrix)
        
        # Get top N recommendations
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        
        recommendations = self.properties_df.iloc[top_indices]
        
        return recommendations.to_dict('records')

# To test this file, you can add:
if __name__ == '__main__':
    recommender = RecommendationService()
    prefs = {'bhk': 3, 'bath': 2, 'price': 100} # 3 BHK, 2 bath, 100 Lakhs budget
    recommendations = recommender.get_recommendations(prefs)
    print(f"Recommendations for a {prefs['bhk']} BHK, {prefs['bath']} bath property with a budget of {prefs['budget']} Lakhs:")
    for prop in recommendations:
        print(f"- {prop['size']} in {prop['location']} for ₹{prop['price']} Lakhs")
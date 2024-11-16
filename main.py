from api_handler import get_viewed_posts, get_liked_posts, get_ratings, get_all_posts,get_all_users
from preprocessing import preprocess_interaction_data, preprocess_video_metadata
from recommendation_model import content_based_recommendation, collaborative_filtering_recommendation

# Fetch and preprocess data
viewed_data = get_viewed_posts()
liked_data = get_liked_posts()
ratings_data = get_ratings()
posts_data = get_all_posts()

# Data Preprocessing
interaction_data = preprocess_interaction_data(viewed_data, liked_data, ratings_data)
video_metadata = preprocess_video_metadata(posts_data)

# Example User for recommendations
user_id = get_all_users()["users"][10]["id"]  # Example user ID
# user_id = 29
print("User ID:", user_id)
# Content-based recommendations
content_recommendations = content_based_recommendation(interaction_data, video_metadata, user_id)
print("Content-Based Recommendations:", content_recommendations)

# Collaborative Filtering recommendations
collab_recommendations = collaborative_filtering_recommendation(interaction_data, user_id)
print("Collaborative Recommendations:", collab_recommendations[0])

from flask import Flask, request, jsonify
from api_handler import get_viewed_posts, get_liked_posts, get_ratings, get_all_posts, get_all_users
from preprocessing import preprocess_interaction_data, preprocess_video_metadata
from recommendation_model import content_based_recommendation, collaborative_filtering_recommendation

app = Flask(__name__)

# Pre-fetch and preprocess data on server start
viewed_data = get_viewed_posts()
liked_data = get_liked_posts()
ratings_data = get_ratings()
posts_data = get_all_posts()

# Preprocess the interaction data and video metadata
interaction_data = preprocess_interaction_data(viewed_data, liked_data, ratings_data)
video_metadata = preprocess_video_metadata(posts_data)

# Flask routes
@app.route('/recommendations/<user_id>  ', methods=['GET'])
def get_recommendations(user_id):
    """
    Endpoint to fetch recommendations for a specific user.
    Example: /recommendations?user_index=10
    """
    try:
        # Get the user index from query params (default is 0)
        user_index = int(user_id)

        
        # Fetch user_id based on the index
        users = get_all_users()["users"]
        if user_index < 0 or user_index >= len(users):
            return jsonify({"error": "Invalid user index"}), 400
        
        user_id = users[user_index]["id"]

        # Generate content-based recommendations
        content_recommendations = content_based_recommendation(interaction_data, video_metadata, user_id)
        
        # Generate collaborative filtering recommendations
        collab_recommendations = collaborative_filtering_recommendation(interaction_data, user_id)
        
        # Construct the response
        response = {
            "user_id": user_id,
            "content_based_recommendations": content_recommendations,
            "collaborative_recommendations": collab_recommendations[:10]  # Limit to top 10 recommendations
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

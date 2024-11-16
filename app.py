from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api_handler import get_viewed_posts, get_liked_posts, get_ratings, get_all_posts, get_all_users
from preprocessing import preprocess_interaction_data, preprocess_video_metadata
from recommendation_model import content_based_recommendation, collaborative_filtering_recommendation
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://127.0.0.1:5000", "http://localhost:5000","http://localhost:3000","https://lyrics-react-flax.vercel.app/"], # Allow your frontend origin
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, Content-Type, etc.)
)
# Pre-fetch and preprocess data on server start
viewed_data = get_viewed_posts()
liked_data = get_liked_posts()
ratings_data = get_ratings()
posts_data = get_all_posts()

# Preprocess the interaction data and video metadata
interaction_data = preprocess_interaction_data(viewed_data, liked_data, ratings_data)
video_metadata = preprocess_video_metadata(posts_data)

# Pydantic models for response data
class RecommendationResponse(BaseModel):
    user_id: int
    content_based_recommendations: list
    collaborative_recommendations: list

@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(user_index: int = 0):
    """
    Endpoint to fetch recommendations for a specific user.
    Example: /recommendations?user_index=10
    """
    try:
        # Fetch user_id based on the index
        users = get_all_users()["users"]
        if user_index < 0 or user_index >= len(users):
            raise HTTPException(status_code=400, detail="Invalid user index")
        
        user_id = users[user_index]["id"]

        # Generate content-based recommendations
        content_recommendations = content_based_recommendation(interaction_data, video_metadata, user_id)
        
        # Generate collaborative filtering recommendations
        collab_recommendations = collaborative_filtering_recommendation(interaction_data, user_id)
        
        # Construct the response
        return RecommendationResponse(
            user_id=user_id,
            content_based_recommendations=content_recommendations,
            collaborative_recommendations=collab_recommendations[:10]  # Limit to top 10 recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api_handler import get_viewed_posts, get_liked_posts, get_ratings, get_all_posts, get_all_users
from preprocessing import preprocess_interaction_data, preprocess_video_metadata
from recommendation_model import content_based_recommendation, collaborative_filtering_recommendation
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
import pandas as pd
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://127.0.0.1:5000", "http://localhost:5000","http://localhost:3000"], # Allow your frontend origin
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

viewed_data = get_viewed_posts()
liked_data = get_liked_posts()
ratings_data = get_ratings()
posts_data = get_all_posts()


interaction_data = preprocess_interaction_data(viewed_data, liked_data, ratings_data)
video_metadata = preprocess_video_metadata(posts_data)

class RecommendationResponse(BaseModel):
    user_id: int
    content_based_recommendations: list
    collaborative_recommendations: list

from fastapi import FastAPI, HTTPException
import logging

logging.basicConfig(level=logging.DEBUG)

@app.get("/recommendations")
async def get_recommendations(user_index: int = 0):
    try:

        logging.debug(f"Received user_index: {user_index}")


        users = get_all_users()["users"]
        if user_index < 0 or user_index >= len(users):
            raise HTTPException(status_code=400, detail="Invalid user index")

        user_id = users[user_index]["id"]

        logging.debug(f"Fetched user_id: {user_id}")

        content_recommendations = content_based_recommendation(interaction_data, video_metadata, user_id)
        logging.debug(f"Raw content_recommendations: {content_recommendations}")


        if isinstance(content_recommendations, pd.DataFrame):
            if content_recommendations.empty:
                logging.error("content_recommendations DataFrame is empty")
                raise HTTPException(status_code=500, detail="Content recommendations are empty")
            content_recommendations = content_recommendations.to_dict(orient='records') 

        if not content_recommendations or not isinstance(content_recommendations, list):
            logging.error("Invalid or empty recommendations received")
            raise HTTPException(status_code=500, detail="Error processing content recommendations")


        if 'posts' not in content_recommendations[0]:
            raise HTTPException(status_code=500, detail="Missing 'posts' in recommendations")

        posts = [item['posts'] for item in content_recommendations]

 
        if not isinstance(posts, list):
            logging.error(f"Expected list of posts but got: {type(posts)}")
            raise HTTPException(status_code=500, detail="Unexpected posts structure")

  
        post_details = []
        for post in posts:
            if isinstance(post, dict) and 'id' in post and 'category' in post:
                post_details.append({
                    "post_id": post['id'],
                    "category": post['category'].get("name", "Unknown"),
                    "description": post.get("post_summary", {}).get("description", "No description available"),  # Safely access description
                    "video_link": post.get("video_link", "No video link available")  # Safely access video link
                })
            else:
                logging.warning(f"Invalid post format: {post}")

        logging.debug(f"Filtered post details: {post_details}")

        collab_recommendations = collaborative_filtering_recommendation(interaction_data, user_id)
        logging.debug(f"Collaborative filtering recommendations: {collab_recommendations}")

        post_details = [{"post_id": int(post["post_id"]), "category": post["category"], "description": post["description"], "video_link": post["video_link"]} for post in post_details]

        return {
            "user_id": int(user_id),
            "content_based_recommendations": post_details, 
            "collaborative_recommendations": collab_recommendations[:10]  
        }
    except Exception as e:
        logging.error(f"Error in getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

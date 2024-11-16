from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def content_based_recommendation(user_interactions, video_metadata, user_id, top_n=10):
    # Extract user history
    # print("user_interactions", user_interactions)
    user_history = user_interactions[user_interactions["id"] == user_id]
    # print("user_history", user_history)
    # print("video_metadata", video_metadata["posts"])
    
    # Extract post_id from 'posts' (assuming it's a dictionary or list of dictionaries)
    video_metadata['post_id'] = video_metadata['posts'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
    user_viewed = video_metadata[video_metadata["post_id"].isin(user_history["id"])]
    # print("user_viewed", user_viewed)
    
    # Ensure tags are in the correct format (strings instead of lists)
    video_metadata['tags'] = video_metadata['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # Combine metadata for similarity
    tfidf = TfidfVectorizer(stop_words='english')
    # print("video_metadata['tags']", video_metadata["tags"])
    
    # Fit and transform the video metadata's tags
    video_tfidf = tfidf.fit_transform(video_metadata)
    
    # Transform the user's viewed videos' tags
    user_tfidf = tfidf.transform(user_viewed)
    
    # Compute similarity between user's viewed videos and all other videos
    similarity = cosine_similarity(user_tfidf, video_tfidf)
    
    # Get top_n recommendations based on similarity
    recommendations = similarity.mean(axis=0).argsort()[-top_n:][::-1]
    return video_metadata.iloc[recommendations]



def collaborative_filtering_recommendation(user_interactions, user_id, top_n=10):
    # Create a user-item matrix
    user_item_matrix = user_interactions.pivot_table(index='id', values='average_rating').fillna(0)
    
    # Compute similarity matrix using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    # print("user_similarity_df", user_similarity_df)
    # Get the target user's similarity scores
    user_sim_scores = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Weighted sum of ratings from similar users
    similar_users = user_sim_scores.index[1:]  # Exclude the user itself
    weights = user_sim_scores[1:].values
    weighted_ratings = np.dot(weights, user_item_matrix.loc[similar_users])
    
    # Recommend top unviewed videos
    user_ratings = user_item_matrix.loc[user_id]
    unviewed_videos = user_ratings[user_ratings == 0].index
    recommendations = sorted([(video, weighted_ratings[idx]) for idx, video in enumerate(unviewed_videos)],
                             key=lambda x: x[1], reverse=True)[:top_n]
    return [video for video, _ in recommendations]
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
    
    video_metadata['post_id'] = video_metadata['posts'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
    user_viewed = video_metadata[video_metadata["post_id"].isin(user_history["id"])]
    # print("user_viewed", user_viewed)
    

    video_metadata['tags'] = video_metadata['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    tfidf = TfidfVectorizer(stop_words='english')
    # print("video_metadata['tags']", video_metadata["tags"])
    
    video_tfidf = tfidf.fit_transform(video_metadata)
    
    user_tfidf = tfidf.transform(user_viewed)
    
    similarity = cosine_similarity(user_tfidf, video_tfidf)
    
    recommendations = similarity.mean(axis=0).argsort()[-top_n:][::-1]
    return video_metadata.iloc[recommendations]



def collaborative_filtering_recommendation(user_interactions, user_id, top_n=10):

    user_item_matrix = user_interactions.pivot_table(index='id', values='average_rating').fillna(0)
    
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    # print("user_similarity_df", user_similarity_df)
    
    user_sim_scores = user_similarity_df[user_id].sort_values(ascending=False)
    
    similar_users = user_sim_scores.index[1:]  
    weights = user_sim_scores[1:].values
    weighted_ratings = np.dot(weights, user_item_matrix.loc[similar_users])
    
    user_ratings = user_item_matrix.loc[user_id]
    unviewed_videos = user_ratings[user_ratings == 0].index
    recommendations = sorted([(video, weighted_ratings[idx]) for idx, video in enumerate(unviewed_videos)],
                             key=lambda x: x[1], reverse=True)[:top_n]
    return [video for video, _ in recommendations]
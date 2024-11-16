import pandas as pd

def preprocess_interaction_data(viewed_data, liked_data, ratings_data):
    viewed_df = pd.DataFrame(viewed_data["posts"])
    liked_df = pd.DataFrame(liked_data["posts"])
    ratings_df = pd.DataFrame(ratings_data["posts"])

    # Print DataFrames for debugging
    print("Viewed DataFrame Columns:", viewed_df.columns)
    print(viewed_df.head())
    
    print("Liked DataFrame Columns:", liked_df.columns)
    print(liked_df.head())
    
    print("Ratings DataFrame Columns:", ratings_df.columns)
    print(ratings_df.head())

    # Adjust columns based on the actual names
    interaction_df = pd.merge(viewed_df[['id', 'username', 'view_count']], liked_df[['id', 'username', 'upvoted']], on=["id", "username"], how="outer")
    interaction_df = pd.merge(interaction_df, ratings_df[['id', 'username', 'average_rating']], on=["id", "username"], how="outer")
    interaction_df.fillna(0, inplace=True)

    return interaction_df


def preprocess_video_metadata(posts_data):
    # Convert the metadata into a DataFrame
    posts_df = pd.DataFrame(posts_data)
    print("Metadata DataFrame Columns:", posts_df.columns)
    print(posts_df.head())
    # Handle any nested structures or lists within 'metadata' and 'tags'
    posts_df['tags'] = posts_df['posts'].apply(lambda x: x.get('tags', []))  # Default to empty list if 'tags' is missing
    
    return posts_df

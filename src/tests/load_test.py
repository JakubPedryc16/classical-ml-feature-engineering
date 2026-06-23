import os

data_path = "data/genres_original"

if os.path.exists(data_path):
    genres = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Data directory found")
    print(f"Number of genres: {len(genres)}")
    print(f"Genres: {genres}")
    
    first_genre = genres[0]
    sample_count = len(os.listdir(os.path.join(data_path, first_genre)))
    print(f"Sample count in '{first_genre}': {sample_count}")
else:
    print("Directory data/genres_original not found. Check the path.")
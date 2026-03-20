import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================
# B. DATASET EXPLORATION
# ============================================

# Sample rating dataset (Users rating Movies)
data = {
    "Movie1": [5, 4, 0, 0, 1],
    "Movie2": [4, 0, 0, 2, 1],
    "Movie3": [0, 0, 5, 4, 0],
    "Movie4": [0, 3, 4, 0, 0],
    "Movie5": [1, 0, 4, 5, 0]
}

ratings = pd.DataFrame(data, index=["User1", "User2", "User3", "User4", "User5"])
print("Ratings Matrix:\n", ratings)

# ============================================
# C. USER-BASED COLLABORATIVE FILTERING
# ============================================

# Cosine similarity between users
user_similarity = pd.DataFrame(
    cosine_similarity(ratings),
    index=ratings.index,
    columns=ratings.index
)

print("\nUser Similarity Matrix:\n", user_similarity)

# Function to recommend movies using User-Based CF
def user_based_recommend(user, top_n=2):
    sim_users = user_similarity[user].sort_values(ascending=False)[1:]  # remove itself
    weighted_ratings = np.zeros(ratings.shape[1])

    for sim_user, sim_score in sim_users.items():
        weighted_ratings += sim_score * ratings.loc[sim_user].values

    recommendations = pd.Series(weighted_ratings, index=ratings.columns)

    # Remove already watched movies
    watched = ratings.loc[user]
    recommendations[watched > 0] = 0

    return recommendations.sort_values(ascending=False).head(top_n)

print("\nUser-Based Recommendations for User1:")
print(user_based_recommend("User1"))

# ============================================
# D. ITEM-BASED COLLABORATIVE FILTERING
# ============================================

# Cosine similarity between items (movies)
item_similarity = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=ratings.columns,
    columns=ratings.columns
)

print("\nItem Similarity Matrix:\n", item_similarity)

# Function to recommend movies using Item-Based CF
def item_based_recommend(user, top_n=2):
    user_ratings = ratings.loc[user]
    scores = pd.Series(0, index=ratings.columns)

    for movie in ratings.columns:
        if user_ratings[movie] > 0:  # if user rated this movie
            scores += item_similarity[movie] * user_ratings[movie]

    # Remove already watched movies
    scores[user_ratings > 0] = 0

    return scores.sort_values(ascending=False).head(top_n)

print("\nItem-Based Recommendations for User1:")
print(item_based_recommend("User1"))

# ============================================
# E. HYBRID RECOMMENDER SYSTEM
# (Combine user-based and item-based scores)
# ============================================

def hybrid_recommend(user, top_n=2, alpha=0.5):
    user_rec = user_based_recommend(user, top_n=ratings.shape[1])
    item_rec = item_based_recommend(user, top_n=ratings.shape[1])

    hybrid_score = alpha * user_rec + (1 - alpha) * item_rec
    hybrid_score = hybrid_score.fillna(0)

    return hybrid_score.sort_values(ascending=False).head(top_n)

print("\nHybrid Recommendations for User1:")
print(hybrid_recommend("User1"))

# ============================================
# F. EVALUATION (Precision, Recall, F1)
# ============================================

# Convert ratings to binary (liked if rating >= 4)
binary_ratings = (ratings >= 4).astype(int)

# Assume we test User1 recommendations
true_items = binary_ratings.loc["User1"]
recommended_items = hybrid_recommend("User1", top_n=2)

pred = pd.Series(0, index=ratings.columns)
pred[recommended_items.index] = 1

precision = precision_score(true_items, pred)
recall = recall_score(true_items, pred)
f1 = f1_score(true_items, pred)

print("\nEvaluation for User1 (Hybrid Model):")
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

orders = [
    ["AAAA", "BBBB", "DDDD"],
    ["AAAA", "CCCC", "EEEE"],
    ["BBBB", "DDDD", "EEEE"],
    ["AAAA", "BBBB"],
    ["CCCC", "EEEE"]
]


# find Matrix
dish_list = list(dish for order in orders for dish in order)
dish_index = {dish: i for i, dish in enumerate(dish_list)}
order_vectors = np.zeros((len(orders), len(dish_list)))

for i, order in enumerate(orders):
    for dish in order:
        order_vectors[i, dish_index[dish]] = 1

# Calculate Similarity
similarity_matrix = cosine_similarity(order_vectors.T)

# Recommendation
def cf_recommend(dish, top_n=3):
    if dish not in dish_index:
        return []
    dish_id = dish_index[dish]
    sorted_indices = np.argsort(-similarity_matrix[dish_id])  # rank by similarity
    return [dish_list[i] for i in sorted_indices[1:top_n+1]]

# Example
print(cf_recommend("AAAA"))

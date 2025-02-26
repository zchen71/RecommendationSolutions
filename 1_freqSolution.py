from collections import defaultdict

# Assume the data can be converted into following sequences
orders = [
    ["AAAA", "BBBB", "DDDD"],
    ["AAAA", "CCCC", "EEEE"],
    ["BBBB", "DDDD", "EEEE"],
    ["AAAA", "BBBB"],
    ["CCCC", "EEEE"]
]

# Count the frequency that has been ordered together
co_occur = defaultdict(lambda: defaultdict(int))

for order in orders:
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            co_occur[order[i]][order[j]] += 1
            co_occur[order[j]][order[i]] += 1

print(co_occur)

# recommendation function
def recommend(dish, top_n=3):
    if dish not in co_occur:
        return []
    sorted_recommendations = sorted(co_occur[dish].items(), key=lambda x: x[1], reverse=True)
    return [dish for dish, _ in sorted_recommendations[:top_n]]

# Recommend methods
print(recommend("AAAA"))  # Recommend dishes that has been ordered with AAAA
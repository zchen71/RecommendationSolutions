from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

orders = [
    ["AAAA", "BBBB", "DDDD"],
    ["AAAA", "CCCC", "EEEE"],
    ["BBBB", "DDDD", "EEEE"],
    ["AAAA", "BBBB"],
    ["CCCC", "EEEE"]
]

# convert to DataFrame
all_dishes = set(dish for order in orders for dish in order)
data = [{dish: (dish in order) for dish in all_dishes} for order in orders]
df = pd.DataFrame(data)

# Run Apriori
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)




def apriori_recommend(dish, top_n=3):
    filtered_rules = rules[rules['antecedents'].apply(lambda x: dish in x)]
    sorted_rules = filtered_rules.sort_values(by='confidence', ascending=False)
    print(sorted_rules)
    return list(sorted_rules['consequents'].explode().unique())[:top_n]

# Example
print(apriori_recommend("AAAA"))

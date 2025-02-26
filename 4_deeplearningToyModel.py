import numpy as np
from tensorflow import keras

# order history information
restaurant_orders = {
    "RestaurantA": [
        ["AAAA", "BBBB", "EEEE"],
        ["AAAA", "CCCC"],
        ["BBBB", "EEEE"],
        ["AAAA", "DDDD", "EEEE"],
        ["CCCC", "DDDD"]
    ],
    "RestaurantB": [
        ["AA", "BB"],
        ["DD", "AA", "EE"],
        ["AA", "CC"]
    ]
}

# One-hot encoding
def build_dish_matrix(restaurant_name):
    orders = restaurant_orders[restaurant_name]
    dish_list = list(set(dish for order in orders for dish in order))  # 去重
    dish_index = {dish: i for i, dish in enumerate(dish_list)}

    order_vectors = np.zeros((len(orders), len(dish_list)))
    for i, order in enumerate(orders):
        for dish in order:
            order_vectors[i, dish_index[dish]] = 1  # one-hot 编码

    return order_vectors, dish_list

# obtain history data
restaurant_name = "RestaurantA"
X, dishes = build_dish_matrix(restaurant_name)
y = X # Self Supervise

# Build Neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(len(dishes),)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(dishes), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X, y, epochs=100, batch_size=2, verbose=0)

# Recommend
def nn_recommend(selected_dishes, top_n=3):
    input_vector = np.zeros((1, len(dishes)))

    # set ordered dishes into 1
    for dish in selected_dishes:
        if dish in dishes:
            input_vector[0, dishes.index(dish)] = 1

    predictions = model.predict(input_vector)[0]  # predict
    sorted_indices = np.argsort(-predictions)  # sort based on predict

    # filter the dishes that has been ordered
    recommended = [dishes[i] for i in sorted_indices if dishes[i] not in selected_dishes][:top_n]
    return recommended

# Record selected dishes
selected_dishes = []

# user order one food
selected_dishes.append("AAAA")
print(f"Order {selected_dishes} recommend: {nn_recommend(selected_dishes)}")

# Order the second food
selected_dishes.append("EEEE")
print(f"order {selected_dishes} recommend: {nn_recommend(selected_dishes)}")

# Order the third food
selected_dishes.append("CCCC")
print(f"order {selected_dishes} recommend: {nn_recommend(selected_dishes)}")

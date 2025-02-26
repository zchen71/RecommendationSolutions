import openai
import json


# ========== 1. 加载餐厅历史数据 ==========
def load_historical_data():
    """模拟餐厅历史订单数据"""
    return {
        "Restaurant_A": [
            ["Kung Pao Chicken", "Rice", "Sweet and Sour Pork"],
            ["Kung Pao Chicken", "Mapo Tofu", "Rice"],
            ["Sweet and Sour Pork", "Rice"],
            ["Braised Pork", "Steamed Bun"],
            ["Mapo Tofu", "Braised Pork"]
        ]
    }


historical_data = load_historical_data()


import openai

# 设置 OpenAI API Key
openai.api_key = "key"

# 定义推荐函数
def generate_recommendation(restaurant_name, selected_dishes):
    """使用 OpenAI GPT-4 生成推荐"""
    restaurant_orders = historical_data.get(restaurant_name, [])
    prompt = f"""
    You are an AI assistant specializing in restaurant menu recommendations.
    Below are past customer orders for {restaurant_name}:
    {json.dumps(restaurant_orders, indent=4)}
    A customer has just ordered: {', '.join(selected_dishes)}.
    Based on historical orders, what are the top 3 dishes to recommend?
    Provide the answer in a concise format: ["Dish 1", "Dish 2", "Dish 3"].
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful restaurant recommendation assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )

    return response.choices[0].message.content

# 示例调用
selected_dishes = ["Kung Pao Chicken"]
recommendations = generate_recommendation("Restaurant_A", selected_dishes)
print(recommendations)

#
# # ========== 2. 通过 OpenAI API 生成推荐 ==========
# def generate_recommendation(restaurant_name, selected_dishes):
#     """使用 OpenAI GPT-4 生成推荐"""
#     restaurant_orders = historical_data.get(restaurant_name, [])
#
#     # 构造 Prompt
#     prompt = f"""
#     You are an AI assistant specializing in restaurant menu recommendations.
#     Below are past customer orders for {restaurant_name}:
#     {json.dumps(restaurant_orders, indent=4)}
#     A customer has just ordered: {', '.join(selected_dishes)}.
#     Based on historical orders, what are the top 3 dishes to recommend?
#     Provide the answer in a concise format: ["Dish 1", "Dish 2", "Dish 3"].
#     """
#
#     # 远程调用 OpenAI GPT-4 API
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7,
#         max_tokens=100
#     )
#
#     return response["choices"][0]["message"]["content"]
#
#
# # ========== 3. 示例调用 ==========
# openai.api_key = "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"  # 替换为你的 OpenAI API Key
# selected_dishes = ["Kung Pao Chicken"]
# recommendations = generate_recommendation("Restaurant_A", selected_dishes)
# print(recommendations)

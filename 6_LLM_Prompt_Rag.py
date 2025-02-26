import json
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== 1. local model installation ==========
MODEL_PATH = "path of the downloaded"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-chat",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_text(prompt):
    """使用本地 DeepSeek LLM 生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ========== 2. fake historical data ==========
historical_data = {
    "Restaurant_A": [
        ["Kung Pao Chicken", "Rice", "Sweet and Sour Pork"],
        ["Kung Pao Chicken", "Mapo Tofu", "Rice"],
        ["Sweet and Sour Pork", "Rice"],
        ["Braised Pork", "Steamed Bun"],
        ["Mapo Tofu", "Braised Pork"]
    ],
    "Restaurant_B": [
        ["Sushi", "Miso Soup"],
        ["Sashimi", "Sushi", "Tempura"],
        ["Sushi", "Ramen"]
    ]
}

# ========== 3. initial data ==========
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

def store_in_chroma(restaurant, orders):
    """存储订单到 ChromaDB"""
    chroma_db = Chroma(persist_directory=f"chroma_{restaurant}", embedding_function=embedding_model)
    for order in orders:
        text_order = " ".join(order)
        chroma_db.add_texts([text_order])
    chroma_db.persist()

# 存储所有餐厅历史订单
for restaurant, orders in historical_data.items():
    store_in_chroma(restaurant, orders)

# ========== 4. Rag + LLM ==========
def generate_recommendation(restaurant_name, selected_dishes):
    """使用 RAG + DeepSeek 生成推荐"""
    # 从 ChromaDB 检索相似订单
    chroma_db = Chroma(persist_directory=f"chroma_{restaurant_name}", embedding_function=embedding_model)
    query = " ".join(selected_dishes)
    similar_orders = chroma_db.similarity_search(query, k=3)

    # 构造 Prompt
    prompt = f"""
    You are an AI assistant specializing in restaurant menu recommendations.
    Below are past customer orders for {restaurant_name}:
    {json.dumps([order.page_content for order in similar_orders], indent=4)}
    A customer has just ordered: {', '.join(selected_dishes)}.
    Based on similar past orders, what are the top 3 dishes to recommend?
    Provide the answer in a concise format: ["Dish 1", "Dish 2", "Dish 3"].
    """

    # 使用本地 DeepSeek LLM 生成推荐
    return generate_text(prompt)


if __name__ == "__main__":
    restaurant_name = "Restaurant_A"
    selected_dishes = ["Kung Pao Chicken"]

    print("🔍 Searching for similar orders...")
    recommendations = generate_recommendation(restaurant_name, selected_dishes)

    print("\n🍽️ Recommended Dishes:")
    print(recommendations)

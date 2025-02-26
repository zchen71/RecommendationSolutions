import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# ========== 1. Loading deepseek ==========
MODEL_PATH = "deepseek-ai/deepseek-llm-7b-chat"  # 也可以替换为本地模型路径
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ========== 2. Define LoRA ==========
lora_config = LoraConfig(
    r=16,  # LoRA 低秩矩阵维度
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 仅微调注意力层
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)


# ========== 3. Loading history data ==========
def load_training_data():
    """加载餐厅历史订单数据"""
    dataset = [
        {
            "input": "Customer ordered: Kung Pao Chicken. What should be recommended next?",
            "output": "Based on previous orders, we recommend: Rice, Sweet and Sour Pork, and Mapo Tofu."
        },
        {
            "input": "Customer ordered: Sushi. What should be recommended next?",
            "output": "Based on previous orders, we recommend: Miso Soup, Tempura, and Sashimi."
        },
        {
            "input": "Customer ordered: Braised Pork. What should be recommended next?",
            "output": "Steamed Bun, Mapo Tofu, and Rice."
        }
    ]
    return dataset


dataset = load_training_data()


# ========== 4. transfer dataset ==========
def format_dataset(dataset):
    """格式化数据集为 Hugging Face 训练格式"""
    inputs = [tokenizer(d["input"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"].squeeze()
              for d in dataset]
    labels = [tokenizer(d["output"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"].squeeze()
              for d in dataset]
    return Dataset.from_dict({"input_ids": inputs, "labels": labels})


train_data = format_dataset(dataset)

# ========== 5. training parameters adjustment ==========
training_args = TrainingArguments(
    output_dir="./deepseek_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_steps=10
)

# ========== 6. Training model ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

trainer.train()

# ========== 7. Save the model ==========
model.save_pretrained("./deepseek_finetuned")
tokenizer.save_pretrained("./deepseek_finetuned")

print("✅ Fine-tuning 完成！模型已保存至 ./deepseek_finetuned")


# ========== 8. Load the new model and recommend ==========
def generate_text(prompt):
    """使用微调后的 DeepSeek 生成推荐"""
    model_finetuned = AutoModelForCausalLM.from_pretrained("./deepseek_finetuned")
    tokenizer_finetuned = AutoTokenizer.from_pretrained("./deepseek_finetuned")

    inputs = tokenizer_finetuned(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model_finetuned.generate(**inputs, max_length=100, temperature=0.7)

    return tokenizer_finetuned.decode(outputs[0], skip_special_tokens=True)


# ========== 9. Run the model ==========
if __name__ == "__main__":
    test_dishes = ["Kung Pao Chicken"]
    prompt = f"Customer ordered: {', '.join(test_dishes)}. What should be recommended next?"

    print("\n🍽️ Recommended Dishes:")
    print(generate_text(prompt))

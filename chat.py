import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

df = pd.read_csv("data.csv")

df['price'] = df['price'].str.replace(',', '').astype(float)

data_dict = df.to_dict(orient="records")

data_context = "\n".join([f"Title: {item['title']}, Price: ₹{item['price']}, Link: {item['link']}" for item in data_dict])

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

def query_llm(prompt):
    input_text = f"Here is the data:\n{data_context}\n\n{prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.long)

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_new_tokens=150,
        num_beams=5,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat_with_data():
    print("Chat with your Amazon data! Type 'exit' to stop.")
    
    while True:
        user_input = input("Ask something about the data: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = query_llm(user_input)
        print(response)

if __name__ == "__main__":
    chat_with_data()
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel


client = OpenAI(api_key="")

def generate_product_prompt(item_description: str, title: str) -> str:
    system_prompt = f"""
    I am sharing the item name and item description of products sold on an e-commerce site. Your task is to determine the most specific word that describes the item, focusing on key characteristics or types from the pictures of the image. For example, if the item is a "car toy," provide "car" as the answer. Please provide a 1-word answer. 
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{item_description} {title}"}
            ],
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


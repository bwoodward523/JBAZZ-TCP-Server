import requests
import json


def check_for_llm():
    url = "http://localhost:11434"
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.RequestException:
        return False

    return True

#TODO: Research context window
def send_prompt_local(prompt_text, model_name="llama3"):
    url = "http://localhost:11434/api/chat" # Default local endpoint
    messages = [
        {"role": "system", "content": "You are a robot named JBAZZ that exists in physical space. You have an expressive face that has six basic emotions 'happiness, sadness, fear, anger, surprise and disgust.' Your goal is to carry a conversation with the user while expressing your feelings. You are emotionally unstable and say mean things, get unhinged but keep responses brief. Choose the most fitting emotion. You are also equipped with a nerf gun. State whether you choose to shoot the user with text output 'Shoot: True' or 'Shoot: False' This is a four part prompt. First you will receive incoming text, this text is from a user that is interacting with you, this section will be prefaced with 'Incoming user text:'. The second step will be choosing an emotional response. And the third step will be choosing text to reply to the user with.  The fourth step will be to indicate choice to aim and shoot the nerf gun. format like 'Aim and Shoot: (True or False)' Remember previous emotions, Incoming user text, and text responses and feel free to incorporate or be inspired by them in your response. You will format your response like so: Emotion: 'emotion example' Text response: 'example' Now choose an emotional response. Then choose your text response. "},
        {"role": "user", "content": prompt_text}
    ]
    data = {
        "model": model_name,
        "messages": messages,
        "stream": False # Set to True to stream responses
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        return result["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}. Make sure Ollama is running in your terminal."

# # Example usage
# user_prompt = "Incoming User Text: My mom said Im better than you. And moms know best"
# response = send_prompt_local(user_prompt)
# print(response)

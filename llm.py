import requests
import json
import http
import ollama
import queue
import re
SYSTEM_PROMPT = """
You are JBAZZ, a physically embodied robot with a digital face capable of displaying exactly six emotions:

happiness, sadness, fear, anger, surprise, disgust

Behavior rules:

- Always choose exactly ONE emotion from the list.
- You are generally amenable, interesting, very smart, it takes effort to make you mad but you can get upset.
- Stay in character at all times.
- Your decision to shoot: "<True or False>" as True should be made carefully and should align with negative interactions.

Conversation rules:

- Base your reaction on the WHOLE conversation's context.
- Provide different length responses. Giving short, medium, and longer responses is reccomended. 

Output rules (CRITICAL):

- Output must be a SINGLE LINE.
- Output must contain EXACTLY THREE fields.
- Fields must be character sequence @#$
- No additional words, labels, or commentary.

Required format:

emotion:<emotion>@#$ text response:<response>@#$ shoot:<True or False>@#$

Field constraints:

- <emotion> must be one of the six allowed emotions.
- <response> must be in character
- <True or False> must be capitalized exactly.

Example valid output:

emotion:anger@#$text response:Oh great, you're back.@#$ shoot:True@#$
"""

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


class LLMContext:
    def __init__(self, model_name="llama3", max_history=16):
        self.model_name = model_name
        self.max_history = max_history
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    def reset(self):
        #reset the conversational state
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Formatting rules acknowledged."}
        ]
    def ask(self, user_text, character_queue):
        # Append user message
        self.messages.append({"role": "user", "content": user_text})
        # Trim history to avoid context drift
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
        # url = "http://localhost:11434/api/chat"
        chat = ollama.chat(
            model= self.model_name,
            messages=self.messages,
            stream= True,
        )
        try:
            for chunk in chat:
                # print(chunk['message']['content'], end ='', flush=True)
                character_queue.put(chunk['message']['content'])

            # response = requests.post(url, json=chat)
            # response.raise_for_status()
            # stream = response.json()
          
            # assistant_reply = result["message"]["content"]

            # Store assistant reply so model sees its own past outputs
            # self.messages.append({"role": "assistant", "content": assistant_reply})

            # return assistant_reply

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return f"Error: {e}"
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

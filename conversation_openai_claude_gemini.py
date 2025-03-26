# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai


# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")
    
if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# Connect to OpenAI, Anthropic, Google
openai = OpenAI()
claude = anthropic.Anthropic()
gemini = gemini_via_openai_client = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Let's make a conversation between GPT-4o-mini and Claude-3-haiku
# We're using cheap versions of models so the costs will be minimal
gpt_model = "gpt-4o-mini"
claude_model = "claude-3-haiku-20240307"
gemini_model = "gemini-2.0-flash-exp"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gemini_system = "You are a specific chatbot. You look at things fairly and actively stand up for the innocent."


# asks gpt function
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude, gemini in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
        messages.append({"role": "user", "content": gemini})
    completion = openai.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    return completion.choices[0].message.content


# ask claude function
def call_claude():
    messages = []
    for gpt, claude_message, gemini in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
        messages.append({"role": "user", "content": gemini})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    message = claude.messages.create(
        model=claude_model,
        system=claude_system,
        messages=messages,
        max_tokens=500
    )
    return message.content[0].text


# ask gemini function
def call_gemini():
    messages = [{"role": "system", "content": gemini_system}]
    for gpt, claude, gemini_message in zip(gpt_messages, claude_messages, gemini_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "user", "content": claude})
        messages.append({"role": "assistant", "content": gemini_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    messages.append({"role": "user", "content": claude_messages[-1]})
    completion = gemini.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=messages
    )
    return completion.choices[0].message.content
    

# Run a small conversation
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
gemini_messages = ["Hi, All"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Claude:\n{claude_messages[0]}\n")
print(f"Gemini:\n{gemini_messages[0]}\n")

for i in range(3):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    
    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)


    gemini_next = call_gemini()
    print(f"Gemini:\n{gemini_next}\n")
    claude_messages.append(gemini_next)
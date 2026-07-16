import re

def reflect(text):
    # Simple pronoun swap (mirroring)
    swaps = {"i":"you","my":"your","me":"you","am":"are","i'm":"you're"}
    for k, v in swaps.items():
        text = re.sub(r'\b' + k + r'\b', v, text, flags=re.I)
    return text

def chatbot(user_input):
    user_input = user_input.lower().strip()
    
    
    patterns = [
        (r"hello|hi|hey", "Hello! How can I assist you with your order or service?"),
        (r"order (.*)", "Let me check your order {0}. Do you have the order number?"),
        (r"([1-9][0-9]+)", "Thank you for providing the order number {0}. I will look into it."),
        (r"not working|broken|issue", "I'm sorry you're having trouble. Can you describe the issue?"),
        (r"bye|goodbye", "Thank you for contacting us. Have a great day!")
    ]
    
    for pattern, template in patterns:
        match = re.search(pattern, user_input)
        if match:
            groups = match.groups()
            if groups:
                reflected = reflect(groups[0])
                return template.format(reflected)
            else:
                return template
    
    return "I'm not sure. Could you tell me about an order, a problem, or say hello?"

# Conversation loop
while True:
    msg = input("You: ")
    reply = chatbot(msg)
    print("Bot:", reply)
    if msg.lower() in ("bye","goodbye"):
        break
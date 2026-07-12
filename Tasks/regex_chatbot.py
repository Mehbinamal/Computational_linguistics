import re

def chatbot(user_input):

    user_input = user_input.lower()

    if re.match(r"hello|hi|hey", user_input):
        return "Hello!"

    elif re.match(r"how are you", user_input):
        return "I'm doing well."

    elif re.match(r"bye", user_input):
        return "Goodbye!"

    else:
        return "I don't understand."

while True:
    msg = input("You: ")

    response = chatbot(msg)

    print("Bot:", response)

    if msg.lower() == "bye":
        break
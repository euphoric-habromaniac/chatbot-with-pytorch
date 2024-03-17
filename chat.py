import random
import json
import torch
import os  # Add the import statement for os module

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_intents(file_path):
    """Load intents data from JSON file."""
    with open(file_path, 'r') as json_data:
        intents = json.load(json_data)
    return intents

def load_model(file_path):
    """Load pre-trained model from file."""
    data = torch.load(file_path)
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, all_words, tags

def get_response(intents, model, all_words, tags, sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)  # Add an extra dimension for batch size
    output = model(X)
    
    print("Output shape:", output.shape)  # Print the shape of the output tensor
    
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I'm sorry, I didn't quite catch that. Could you please rephrase?"



def main():
    """Main function to run the chatbot."""
    # Construct absolute path to intents.json
    intents_file = os.path.join(os.path.dirname(__file__), 'intents.json')

    intents = load_intents(intents_file)
    model, all_words, tags = load_model("data.pth")
    bot_name = "GamingBot"

    print(f"Welcome to our Gaming Cafe! I'm {bot_name}, your gaming assistant. How can I assist you today? (type 'quit' to exit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = get_response(intents, model, all_words, tags, user_input)
        print(f"{bot_name}: {response}")

if __name__ == "__main__":
    main()

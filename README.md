# Gaming Cafe Chatbot

This is a simple chatbot designed for a gaming cafe. The chatbot assists users by providing information about available games, payment options, opening hours, and more.

## Features

- **Greeting:** Responds to greetings from users.
- **Menu Inquiry:** Provides information about available games and genres.
- **Payment Information:** Answers queries about payment options and costs.
- **Opening Hours:** Provides details about the cafe's opening hours.
- **Goodbye:** Bids farewell to users when they exit the conversation.

## Requirements

- Python 3.x
- PyTorch
- NLTK

## Installation

1. Clone the repository:

https://github.com/euphoric-habromaniac/chatbot-with-pytorch

2. Install required dependencies:
   1. download pytorch from ```pytorch.org```
   2. download nltk with ```pip install nltk```

## Usage

Run train.py then chat.py

## Project Structure

- `chat.py`: Main script to run the chatbot and interact with users.
- `train.py`: Script to train the chatbot model.
- `model.py`: Definition of the chatbot's neural network model.
- `nltk_utils.py`: Utility functions for tokenizing and preprocessing text data.
- `intents.json`: JSON file containing predefined intents for the chatbot.
- `data.pth`: Saved PyTorch model file after training.
- `README.md`: Documentation file (you're reading it right now).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to help improve this project.

## License

This project is licensed under the [MIT License](LICENSE).

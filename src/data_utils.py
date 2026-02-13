from datasets import load_dataset

def load_training_data():
    """
    Loads the training data. Training data is a large string containing text from first 1000 examples

    @return a string containing the training data
    """
    multilingual = load_dataset("allenai/c4", "multilingual", streaming=True)

    data = [] # List of strings
    for i, example in enumerate(multilingual['train']['text']):
        data.append(example)
        # Limiting to 1000 for now to avoid grabbing whole dataset, should be plenty
        if i >= 1000:
            break
    return ' '.join(data)

if __name__ == '__main__':
    training_data = load_training_data()
    print(training_data[:200])
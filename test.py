import torch
from create_model import create_model
from transformers import AutoTokenizer












def test():
    device = torch.device("cuda:0")

    # Create a new model
    Model = create_model()

    # Load in the pretrained model
    Model.load_state_dict(torch.load("models/model_12000.pt"))

    # Load in the state for the tokenizer
    Model.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

    # Move the model to the GPU and set it to evaluation mode
    Model = Model.to(device)
    Model.eval()

    # Prompts for the model
    prompts = [
        "Me: How are you\nYou: I",
        "My name is Thomas and my main",
        "I am a student at the University of",
        "The weather is nice today and I am going to"
    ]

    # Iterate to get 10 outputs
    for output in range(0, 100):
        # Get the next output form the model
        output = Model.forward_pred(prompts)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [Model.tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts)):
            prompts[i] += output_tokens[i]

    print(prompts)



if __name__ == "__main__":
    test()
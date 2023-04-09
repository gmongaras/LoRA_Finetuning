import torch
from create_model import create_model













def test():
    # Create the model form GPT-2
    Model = create_model()

    # Prompts for the model
    prompts = [
        "My name is Thomas and my main",
        "I am a student at the University of",
        "The weather is nice today and I am going to"
    ]

    # Iterate to get 10 outputs
    for output in range(0, 100):
        # Get the next output form the model
        output = Model(prompts)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [Model.tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts)):
            prompts[i] += output_tokens[i]

    print()



if __name__ == "__main__":
    test()
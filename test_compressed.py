import torch
from model import Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from GPT2_LoRA import GPT2_config_LoRA
from create_model import create_model
from LoRA_to_Model import LoRA_to_Model












def test():
    # # Load in the base GPT2 model
    # model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    # # Load in the state for the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

    # # Create a new model
    # model = Model(GPT2_config_LoRA(), tokenizer, model.transformer.wte, model.transformer.wpe, list(model.transformer.h), model.transformer.ln_f, model.lm_head)

    # # Load in the pretrained model
    # model.load_state_dict(torch.load("models/model_1900_compressed.pt"))



    device = torch.device("cuda:0")

    # Create a new model
    model = create_model().eval()

    # Load in the pretrained model
    model.load_state_dict(torch.load("models/model_12000.pt"))

    # Load in the state for the tokenizer
    model.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")


    # Convert the model from LoRA to a compressed model
    new_model = LoRA_to_Model(model)

    # Delete the old model
    del model





    # Set the model to eval mode and put it on the GPU
    new_model = new_model.to(device)
    new_model.eval()

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
        output = new_model.forward_pred(prompts)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [new_model.tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts)):
            prompts[i] += output_tokens[i]

    print(prompts)



if __name__ == "__main__":
    test()
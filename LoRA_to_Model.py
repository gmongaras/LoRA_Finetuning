import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from GPT2 import GPT2Block, GPT2_config, Conv1D
from GPT2_LoRA import GPT2Block_LoRA, GPT2_config_LoRA
from model import Model
from create_model import create_model








def LoRA_to_Model(model):
    # Get the model configuration file
    config = model.config

    # New transformer layers in the model
    layers = list()

    # Iterate over all transformer layers in the model
    params = list(model.layers)

    i = 0
    for i in range(0, len(params)):
        param = params[i]

        # If the parameter is a GPT2 LoRA block, re-encode it as a GPT2 block
        if type(param) != nn.LayerNorm:
            # A, B, and C matrices
            K_A = param.attn.K_A
            Q_A = param.attn.Q_A
            V_A = param.attn.V_A
            W_A = param.attn.W_A
            K_B = param.attn.K_B
            Q_B = param.attn.Q_B
            V_B = param.attn.V_B
            W_B = param.attn.W_B
            K_C = param.attn.K_C
            Q_C = param.attn.Q_C
            V_C = param.attn.V_C
            W_C = param.attn.W_C

            # Normal QKV matrices
            c_attn = param.attn.c_attn
            c_attn_weight = c_attn.weight
            c_attn_bias = c_attn.bias
            
            # Split the c_attn weight and bias into Q, K, and V
            Q_weight, K_weight, V_weight = c_attn_weight.chunk(3, dim=-1)
            Q_bias, K_bias, V_bias = c_attn_bias.chunk(3, dim=-1)

            # Compress the QKV with the equation h = C(WX+b)+XAB = X[(CW+AB) + Cb]
            # Where (CW+AB) is the new weight and Cb is the new bias
            Q_weight = (Q_weight@Q_C)+(Q_A@Q_B)
            Q_bias = Q_bias@Q_C
            K_weight = (K_weight@K_C)+(K_A@K_B)
            K_bias = K_bias@K_C
            V_weight = (V_weight@V_C)+(V_A@V_B)
            V_bias = V_bias@V_C

            # Combine the QKV matrices into a single matrix
            c_attn_weight = torch.cat((Q_weight, K_weight, V_weight), dim=-1).squeeze()
            c_attn_bias = torch.cat((Q_bias, K_bias, V_bias), dim=-1).squeeze()

            # Create a new conv1d layer for the attention
            c_attn = Conv1D(c_attn_weight.shape[1], c_attn_weight.shape[0])
            c_attn.weight.data = c_attn_weight
            c_attn.bias.data = c_attn_bias



            # Now we need to create the output projection matrix using
            # the exact same method
            c_proj = param.attn.c_proj
            c_proj_weight = c_proj.weight
            c_proj_bias = c_proj.bias

            # Compress the projection matrix with the equation h = W(CX)+b+XAB = X[(CW+AB) + b]
            c_proj_weight = ((W_C@c_proj_weight)+(W_A@W_B)).squeeze()
            c_proj_bias = c_proj_bias.squeeze()

            # Create a new conv1d layer for the output projection
            c_proj = Conv1D(c_proj_weight.shape[1], c_proj_weight.shape[0])
            c_proj.weight.data = c_proj_weight
            c_proj.bias.data = c_proj_bias




            # Create a new GPT2 layer
            blk = GPT2Block(config, i)

            # Copy the weights from the original model to the new layer
            blk.attn.c_attn = c_attn
            blk.attn.c_proj = c_proj
            blk.ln_1.weight.data = param.ln_1.weight
            blk.ln_1.bias.data = param.ln_1.bias
            blk.mlp.c_fc.weight.data = param.mlp.c_fc.weight
            blk.mlp.c_fc.bias.data = param.mlp.c_fc.bias
            blk.mlp.c_proj.weight.data = param.mlp.c_proj.weight
            blk.mlp.c_proj.bias.data = param.mlp.c_proj.bias
            blk.ln_2.weight.data = param.ln_2.weight
            blk.ln_2.bias.data = param.ln_2.bias

            # Save the new layer to the list of layers
            layers.append(blk)

        else:
            layers.append(param)



    

    # Create a new model
    new_model = Model(config, model.tokenizer, model.token_embeddings, model.position_embeddings, layers, model.head, config.n_head, int(config.hidden_size*config.hidden_size_scale), config.r)

    

    # Return the new model
    return new_model





        






if __name__ == "__main__":
    # Create a new model
    model = create_model().eval()

    # Load in the pretrained model
    model.load_state_dict(torch.load("models/model_1900.pt"))

    # Load in the state for the tokenizer
    model.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")


    

    # Convert the model from LoRA to a compressed model
    new_model = LoRA_to_Model(model)

    # Save the state of the new model
    torch.save(new_model.state_dict(), "models/model_1900_compressed.pt")




    # Put the model in eval mode 
    new_model = new_model.eval()

    # Prompts for the model
    prompts = [
        "My name is Thomas and my main",
        "I am a student at the University of",
        "The weather is nice today and I am going to"
    ]
    prompts_new = [
        "My name is Thomas and my main",
        "I am a student at the University of",
        "The weather is nice today and I am going to"
    ]

    # Iterate to get 10 outputs from the old model
    for output in range(0, 10):
        # Get the next output form the model
        output = model.forward_pred(prompts)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [new_model.tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts)):
            prompts[i] += output_tokens[i]

    # Iterate to get 10 outputs from the new model
    for output in range(0, 10):
        # Get the next output form the model
        output = new_model.forward_pred(prompts_new)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [new_model.tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts_new)):
            prompts_new[i] += output_tokens[i]

    for t in range(0, len(prompts)):
        assert prompts[t] == prompts_new[t]
        print("Old Model: ", prompts[t])
        print("New Model: ", prompts_new[t])
import torch
from torch import nn
from model import Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from GPT2_LoRA import GPT2Block_LoRA, GPT2_config_LoRA













def create_model():
    # Create a new config file
    config = GPT2_config_LoRA()

    # Params
    d_tilde_scale = int(config.hidden_size*config.hidden_size_scale)
    r = config.r

    # Load in the pretrained GPT2 model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    # How many heads are there?
    n_heads = model.config.n_head



    # Get the embeddings layers for the model
    token_embeddings = model.transformer.wte
    position_embeddings = model.transformer.wpe

    # Output layer normalization
    output_ln = model.transformer.ln_f

    # Get the head of the model
    head = model.lm_head

    


    # Transformer layers in the model
    layers = list()

    # Iterate over all transformer layers in the model and make them actually usable
    params = list(model.transformer.h)

    i = 0
    for i in range(0, len(params)):
        param = params[i]

        # If the parameter is a GPT3 block, re-encode it as a LoRA block
        if type(param) != nn.LayerNorm:
            # Get the attention weights
            attn_weight = param.attn.c_attn.weight
            attn_bias = param.attn.c_attn.bias

            # Get the weight and bias for the output projection
            W_weight = param.attn.c_proj.weight
            W_bias = param.attn.c_proj.bias

            # Create a new GPT3 layer
            blk = GPT2Block_LoRA(config, i)
            # config = GPT2_config()
            # blk = GPT2Block(config, i)

            # Copy the weights from the original model to the new layer
            blk.attn.c_attn.weight.data = attn_weight
            blk.attn.c_attn.bias.data = attn_bias
            blk.attn.c_proj.weight.data = W_weight
            blk.attn.c_proj.bias.data = W_bias
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

    # Add layer normalization to the end of the transformer stack
    layers.append(output_ln)



    model = Model(tokenizer, token_embeddings, position_embeddings, layers, head, n_heads, d_tilde_scale, r)
    model = model.eval()

    return model
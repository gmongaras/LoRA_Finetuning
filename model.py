import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from GPT2 import GPT2Block, GPT2_config
from GPT2_LoRA import GPT2Block_LoRA, GPT2_config_LoRA





class Model(nn.Module):
    # tokenizer: HuggingFace tokenizer to tokenize the input text
    # token_embeddings: nn.Embedding layer to embed tokens from vocab index to embedding vector
    # position_embeddings: nn.Embedding layer to embed positions
    # layers: list of Transformer_LoRA layers
    # head: nn.Linear layer to convert the output of the transformer layers to logits
    # n_heads: number of heads in the transformer layers
    # d_tilde_scale: factor to scale the attention in the transformer layers (should be < 1)
    # r: Middle embedding dimension for the LoRA layers
    def __init__(self, tokenizer, token_embeddings, position_embeddings, layers, head, n_heads, d_tilde_scale=0.5, r=1):
        super(Model, self).__init__()

        # Store the model information
        self.tokenizer = tokenizer
        self.token_embeddings = token_embeddings
        self.position_embeddings = position_embeddings
        self.layers = nn.Sequential(*layers)
        self.head = head

        # The padding token will be at the end of the vocab
        self.pad_token_id = tokenizer.vocab_size - 1

    
    def forward(self, X):
        # Encode the text into a tensor of token indices
        text = []
        max_len = 0
        for x in X:
            t = self.tokenizer.encode(x, return_tensors='pt')[0]
            text.append(t)
            max_len = max(max_len, t.shape[0])

        # Pad the text to the longest length
        for i in range(len(text)):
            text[i] = torch.cat((text[i], torch.full((max_len - text[i].shape[0],), self.pad_token_id).long()), dim=0)

        # Stack the text tensors
        X = torch.stack(text, dim=0)

        # Get a padding mask to mask any padding that was added
        padding_mask = X == self.pad_token_id
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.repeat(1, 1, X.shape[1], 1)
        padding_mask = padding_mask.float() * -1e10

        # Embed the tokens and add positional encodings
        X = self.token_embeddings(X)
        X += self.position_embeddings(torch.arange(0, X.shape[1]).unsqueeze(0).long())

        # Put the embeddings through the transformer stack
        for layer in self.layers:
            if type(layer) != nn.LayerNorm:
                X = layer(X, attention_mask=padding_mask)[0]
            else:
                X = layer(X)

        # Get the output logits
        X = self.head(X).softmax(dim=-1)

        # Get the first instance of a nonzero value in the padding
        # mask to get the length of the text
        lengths = (padding_mask != 0).int().argmax(-1)[:, 0, 0]-1

        # Get the logits for the output token for each batchitem
        output = torch.stack([X[i, lengths[i], :] for i in range(X.shape[0])], dim=0)

        return output





        






if __name__ == "__main__":
    # Params
    d_tilde_scale = 0.5
    r = 1

    # Load in the pretrained GPT2 model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    # How many heads are there?
    n_heads = model.config.n_head



    # Get the transformer layers of the model
    transformer_layers = model.transformer

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

            # Split the weights and biases into Q, K, and V
            Q_weight, K_weight, V_weight = attn_weight.chunk(3, dim=-1)
            Q_bias, K_bias, V_bias = attn_bias.chunk(3, dim=-1)

            # Get the weight and bias for the output projection
            W_weight = param.attn.c_proj.weight
            W_bias = param.attn.c_proj.bias

            # Get the number of features
            d_model = Q_weight.shape[1]

            # Create a new GPT3 layer
            config = GPT2_config_LoRA()
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

    # Prompts for the model
    prompts = [
        "My name is Thomas and my main",
        "I am a student at the University of",
        "The weather is nice today and I am going to"
    ]

    # Iterate to get 10 outputs
    for output in range(0, 100):
        # Get the next output form the model
        output = model(prompts)

        # Get the next output tokens
        output_tokens = torch.argmax(output, dim=-1)
        output_tokens = [tokenizer.decode(t.tolist()) for t in output_tokens]

        # Add the output tokens to the prompt
        for i in range(len(prompts)):
            prompts[i] += output_tokens[i]

    print()
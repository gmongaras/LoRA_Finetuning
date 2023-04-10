import torch
from create_model import create_model
import lzma

from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm






def tokenize(element, tokenizer, context_length):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}






def finetune():
    # Create the model form GPT-2
    Model = create_model().train().cuda()

    # Load in the data
    data = []
    with lzma.open('data/1.xz', mode='rt', encoding='utf-8') as f:
        # Read in all the lines from the file
        lines = f.readlines()

        # Remove any newlines
        lines = [line.replace("\n", "").replace("\x00", "") for line in lines]
        lines = [line for line in lines if len(line) > 0]

        data += lines

    # Join all the data together
    data = "/n".join(data)

    # Write the data to files    
    with open("data/1.txt", "w", encoding='utf-8') as f:
        f.write(data[:100000000])
    with open("data/2.txt", "w", encoding='utf-8') as f:
        f.write(data[1000000000:1500000000])






    # Load the data
    data = load_dataset("text", data_files={"train": ["data/1.txt"], "test": "data/2.txt"}, sample_by="document")

    # Tokenize a subset of the text
    tokenized_data = data.map(
        tokenize, 
        batched=True,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": Model.tokenizer, "context_length": Model.config_file.max_position_embeddings}
    )








    # Collator for easy data loading
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=Model.tokenizer, mlm=False,
    )

    # Load in the train/test data
    tokenized_data.set_format("torch")
    train_dataloader = DataLoader(tokenized_data["train"], batch_size=6, shuffle=True)
    eval_dataloader = DataLoader(tokenized_data["test"], batch_size=4)

    # List of paramters not counting the W pre-trained paramters
    parameters = [p for n, p in Model.named_parameters() if "_A" in n or "_B" in n or "_C" in n]
    # Optimizer for the model
    lr = 5e-4
    optimizer = AdamW(parameters, lr=lr)

    # # We can make the training fp16
    accelerator = Accelerator(mixed_precision="fp16") # Can also be fp8
    Model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        Model, optimizer, train_dataloader, eval_dataloader
    )


    # Some training params
    gradient_accumulation_steps = 1
    eval_steps = 100
    num_train_epochs = 10
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    samples_per_step = 1
    output_dir = "outputs/"

    def loss_funct(inputs, logits):
        # Shift so that tokens < n predict n
        shift_labels = inputs[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
    
    def evaluate():
        Model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = Model(batch["input_ids"], labels=batch["input_ids"])

            losses.append(accelerator.gather(outputs.loss))
        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()
    

    # Start training
    Model.train()
    completed_steps = 0
    for epoch in range(num_train_epochs):
        # Iterate over entire batch
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=min(len(train_dataloader)-1, num_training_steps)
        ):
            # Get output from model
            logits = Model(batch["input_ids"])

            # Get CCE loss
            loss = loss_funct(batch["input_ids"], logits)

            # Every 100 steps, print some info
            if step % 100 == 0:
                accelerator.print(
                    {
                        "lr": lr,
                        "samples": step * samples_per_step,
                        "steps": completed_steps,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                    }
                )

            # Backprop the loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            # Update the model
            if step % gradient_accumulation_steps == 0:
                # accelerator.clip_grad_norm_(Model.parameters(), 1.0)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            # Every so often, evaluate the model
            if (completed_steps % (eval_steps * gradient_accumulation_steps)) == 0:
                tqdm.write(str(round(loss.detach().cpu().item(), 4)))
                # eval_loss, perplexity = evaluate()
                # accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                # Model.train()
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(Model)
                # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                # if accelerator.is_main_process:
                #     Model.tokenizer.save_pretrained(output_dir)



    # # Training arguments
    # training_args = TrainingArguments(
    #     output_dir="./gpt2-gerchef", #The output directory
    #     overwrite_output_dir=True, #overwrite the content of the output directory
    #     num_train_epochs=3, # number of training epochs
    #     per_device_train_batch_size=2, # batch size for training
    #     per_device_eval_batch_size=64,  # batch size for evaluation
    #     eval_steps = 400, # Number of update steps between two evaluations.
    #     save_steps=800, # after # steps model is saved 
    #     warmup_steps=500,# number of warmup steps for learning rate scheduler
    #     prediction_loss_only=False,
    #     remove_unused_columns=False
    # )

    # # Create the trainer
    # trainer = Trainer(
    #     model=Model,
    #     args=training_args,
    #     train_dataset=train_dataloader,
    #     eval_dataset=eval_dataloader,
    #     data_collator=data_collator,
    # )

    # # Train the model
    # trainer.train()
    # trainer.save_model()







    # Put model in eval mode and on CPU
    Model.eval()
    Model = Model.to("cpu")

    # Prompts for the model
    prompts = [
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
    finetune()
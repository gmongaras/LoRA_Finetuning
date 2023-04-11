import lzma
import random




def load_data():
    train_name = "data/train.txt"
    eval_name = "data/eval.txt"
    eval_lines = 100000

    filenames = ["data/1.xz", "data/2.xz"]

    # Load in the data
    data = []
    for file in filenames:
        with lzma.open(file, mode='rt', encoding='utf-8') as f:
            # Read in all the lines from the file
            lines = f.readlines()

            # Remove any newlines
            lines = [line.replace("\n", "").replace("\x00", "") for line in lines]
            lines = [line for line in lines if len(line) > 0]

            data += lines

    # Shuffle the data
    random.shuffle(data)

    # Split data into train/eval
    eval_data = "\n".join(data[:eval_lines])
    train_data = "\n".join(data[eval_lines:])

    # Write the data to files    
    with open(train_name, "w", encoding='utf-8') as f:
        f.write(train_data)
    with open(eval_name, "w", encoding='utf-8') as f:
        f.write(eval_data)






if __name__ == "__main__":
    load_data()
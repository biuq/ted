
from io import BytesIO
import requests
import pandas as pd

jokes_url = 'https://huggingface.co/datasets/ysharma/short_jokes/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true'
oasst2_train_url = 'https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true'
oasst2_val_url = 'https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet?download=true'
alpaca_gpt4_url = 'https://huggingface.co/datasets/vicgalle/alpaca-gpt4/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true'
tiny_shakespeare_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
tiny_stories_train_url = 'https://huggingface.co/datasets/noanabeshima/TinyStoriesV2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true'
tiny_stories_train_2_url = 'https://huggingface.co/datasets/noanabeshima/TinyStoriesV2/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet?download=true'
tiny_stories_val_url = 'https://huggingface.co/datasets/noanabeshima/TinyStoriesV2/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet?download=true'

def load_dataset(name: str) -> tuple[str, str]:
    return {
        'tiny_stories': load_tiny_stories_train_val
    }[name]()

def load_jokes(delim='🥁\n'):
    response = requests.get(jokes_url)
    parquet_file = BytesIO(response.content)
    df = pd.read_parquet(parquet_file)

    def format_row(row):
        return f"💬\n{row['Joke']}\n"

    formatted_text = delim.join(df.apply(format_row, axis=1))
    return formatted_text

def load_oasst2(train: bool = True, prompter_start='🎤\n', prompter_end='✉\n', assistant_start='🤖\n', assistant_end='🖂\n'):
    response = requests.get(oasst2_train_url if train else oasst2_val_url)
    parquet_file = BytesIO(response.content)
    df = pd.read_parquet(parquet_file)

    def format_row(row):
        if row['role'] == 'prompter':
            return f"{prompter_start}{row['text']}{prompter_end}"
        return f"{assistant_start}{row['text']}{assistant_end}"

    formatted_text = ''.join(df.apply(format_row, axis=1))
    return formatted_text

def load_alpaca(prompter_start='🎤\n', prompter_end='✉\n', assistant_start='🤖\n', assistant_end='🖂\n'):
    response = requests.get(alpaca_gpt4_url)
    parquet_file = BytesIO(response.content)
    df = pd.read_parquet(parquet_file)

    def format_row(row):
        if len(row["input"]) > 0:
            return f"{prompter_start}{row['instruction']}\n{row['input']}{prompter_end}{assistant_start}{row['output']}{assistant_end}"
        return f"{prompter_start}{row['instruction']}{prompter_end}{assistant_start}{row['output']}{assistant_end}"

    formatted_text = ''.join(df.apply(format_row, axis=1))
    return formatted_text

def load_tiny_stories(train=True, separator='\n🛑\n'):
    response = requests.get(tiny_stories_train_2_url if train else tiny_stories_val_url)

    parquet_file = BytesIO(response.content)
    df = pd.read_parquet(parquet_file)

    def format_row(row):
        return row['text']
    
    formatted_text = separator.join(df.apply(format_row, axis=1))
    return formatted_text

def load_tiny_shake_train_val():
    response = requests.get(tiny_shakespeare_url)
    text_data = response.text
    n = int(0.9*len(text_data))
    train_data = text_data[:n]
    val_data = text_data[n:]
    return train_data, val_data

def load_tiny_stories_train_val():
    train = load_tiny_stories()
    val = load_tiny_stories(train=False)
    return train, val

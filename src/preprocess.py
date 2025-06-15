import re
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and filter dialogue lines"""
    df = pd.read_csv(csv_path)
    df = df[df['PlayerLine'].notna()]
    df['Player'] = df['Player'].fillna('')
    mask = df['Player'].str.contains(r'^(ACT|SCENE)', regex=True)
    df = df[~mask]
    return df


def clean_text(txt: str) -> str:
    """Remove bracketed directions, unwanted chars, lowercase and collapse whitespace"""
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"\(.*?\)", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s\.,;'-]", "", txt)
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def preprocess_csv(csv_path: str,
                   seq_len: int = 20,
                   out_dir: str = '../data/processed') -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    # Load and clean CSV
    df = load_csv(csv_path)
    combined = ' '.join(df['PlayerLine'].astype(str).tolist())
    cleaned = clean_text(combined)

    # Tokenization
    nltk_tokens = cleaned.split()  # word-level split for clarity
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned])
    with open(f"{out_dir}/tokenizer.pkl", 'wb') as f:
        pickle.dump(tokenizer, f)

    # Convert to integer list
    token_list = tokenizer.texts_to_sequences([cleaned])[0]

    # Sequence generation
    sequences = []
    for i in range(seq_len, len(token_list)):
        seq = token_list[i-seq_len:i+1]
        sequences.append(seq)
    sequences = pad_sequences(sequences, maxlen=seq_len+1, padding='pre')
    X, y = sequences[:, :-1], sequences[:, -1]

    # Save arrays
    np.save(f"{out_dir}/data_X.npy", X)
    np.save(f"{out_dir}/data_y.npy", y)
    return X, y, tokenizer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess Shakespeare CSV with fixed sequence length')
    parser.add_argument('--csv_path', default='../data/raw/shakespeare.csv')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--out_dir', default='../data/processed')
    args = parser.parse_args()
    X, y, tokenizer = preprocess_csv(args.csv_path, args.seq_len, args.out_dir)
    print(f"Preprocessed: X shape {X.shape}, y shape {y.shape}")

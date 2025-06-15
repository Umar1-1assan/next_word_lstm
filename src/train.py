
import argparse
import numpy as np
import pickle
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_model

def main(args):
    # Load data
    X = np.load(args.x_path)
    y = np.load(args.y_path)
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1
    seq_length = X.shape[1]

    # Build model
    model = build_model(vocab_size, seq_length,
                        embed_dim=args.embed_dim,
                        lstm_units=args.lstm_units,
                        dropout_rate=args.dropout)

    optimizer = Adam(learning_rate=args.lr) if args.opt == 'adam' else RMSprop(learning_rate=args.lr)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(args.save_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=[checkpoint, early]
    )
    # Optionally save final model
    model.save(args.save_final)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM next-word model')
    parser.add_argument('--x_path', default='../data/processed/data_X.npy')
    parser.add_argument('--y_path', default='../data/processed/data_y.npy')
    parser.add_argument('--tokenizer_path', default='../data/processed/tokenizer.pkl')
    parser.add_argument('--save_path', default='../models/best_model.h5')
    parser.add_argument('--save_final', default='../models/final_model.h5')
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--lstm_units', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--opt', choices=['adam','rmsprop'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()
    main(args)

from fastapi import FastAPI
from pydantic import BaseModel
import chess
from fastapi.middleware.cors import CORSMiddleware
import random
from typing import Optional
import numpy as np
from IPython.display import SVG, display
import pandas as pd

ai_move = None  # グローバル変数を宣言

# Great! We can now use this function to get any board state ready for our model!
# We'll use pandas to pull in all our training data
train_df = pd.read_csv('train.csv', index_col='id')

# We'll only use the first 10000 examples so things run fast,
# but you'll get better performance if you remove this line
train_df = train_df[:10000]

# We'll also grab the last 1000 examples as a validation set
val_df = train_df[-1000:]
train_df.head()

def one_hot_encode_peice(piece):
    pieces = list('rnbqkpRNBQKP.')
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr

one_hot_encode_peice('b')
def encode_board(board):
    # first lets turn the board into a string
    board_str = str(board)
    # then lets remove all the spaces
    board_str = board_str.replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            row_list.append(one_hot_encode_peice(piece))
        board_list.append(row_list)
    return np.array(board_list)
encode_board(chess.Board())

def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

# We'll stack all our encoded boards into a single numpy array
X_train = np.stack(train_df['board'].apply(encode_fen_string))
y_train = train_df['black_score']


X_val = np.stack(val_df['board'].apply(encode_fen_string))
y_val = val_df['black_score']


# Let's test on the starting board
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
X_train = X_train.reshape(-1, 8, 8, 13)
X_val = X_val.reshape(-1, 8, 8, 13)

# With the Keras Sequential model we can stack neural network layers together
model = Sequential([
    Flatten(input_shape=(1, 8, 8, 13)),
    Dense(128, activation='relu'),
    Dense(1),
])
model.compile(
    optimizer='rmsprop',
    loss='mean_squared_error')

X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
# To test things out, let's train for 20 epochs and see how our model is doing
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlayGameRequest(BaseModel):
    lastMove: Optional[str] = None
    fen: str

def play_random(fen):
    board = chess.Board(fen=fen)
    move = random.choice(list(board.legal_moves))
    return str(move)

def play_nn(fen, show_move_evaluations=False):
    board = chess.Board(fen=fen)

    moves = []
    input_vectors = []
    for move in board.legal_moves:
        candidate_board = board.copy()
        candidate_board.push(move)
        moves.append(move)
        input_vectors.append(encode_board(str(candidate_board)).astype(np.int32))
    
    input_vectors = np.array(input_vectors)
    input_vectors = input_vectors.reshape(-1, 1, 8, 8, 13)
    
    scores = model.predict(input_vectors, verbose=0)
    
    if board.turn == chess.BLACK:
        index_of_best_move = np.argmax(scores)
    else:
        index_of_best_move = np.argmax(-scores)

    if show_move_evaluations:
        print(zip(moves, scores))
        
    best_move = moves[index_of_best_move]

    return str(best_move)
        
    best_move = moves[index_of_best_move]

    # Now we turn our move into a string, return it and call it a day!
    return str(best_move)
def play_game(ai_function, last_move, fen):
    global ai_move
    board = chess.Board(fen)
    while board.outcome() is None:
        display(SVG(board._repr_svg_()))
        if board.turn == chess.WHITE:
            board.push_san(last_move)
        elif board.turn == chess.BLACK:
            ai_move = ai_function(board.fen())
            print(f'AI move: {ai_move}')
            board.push_san(ai_move)
            return ai_move
    print(board.outcome())

@app.post("/play_game")
async def play(request: PlayGameRequest):
    last_move = request.lastMove
    fen = request.fen
    print("last_move", last_move)
    ai_move = play_game(play_nn, last_move, fen)
    return ai_move
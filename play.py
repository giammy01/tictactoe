import torch
from tictactoe import TicTacToeBoard
from modelttt import TicTacToeNet
from trainttt import MCTS

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\n")
    for i in range(3):
        # Map the internal array to visual symbols
        row = [symbols[board.board[i*3 + j]] for j in range(3)]
        # If the square is empty, show the number 0-8 so the human knows what to type
        row = [str(i*3 + j) if r == ' ' else r for j, r in enumerate(row)]
        
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 2:
            print("---+---+---")
    print("\n")

def play():
    # 1. Load the trained brain
    model = TicTacToeNet()
    try:
        model.load_state_dict(torch.load("tictactoe_model.pth"))
        model.eval() # Set to evaluation mode (turns off training behaviors)
        print("Successfully loaded AI brain!")
    except FileNotFoundError:
        print("Error: Could not find tictactoe_model.pth.")
        return

    # 2. Initialize the Board and MCTS
    board = TicTacToeBoard()
    # We lower the exploration constant (c_puct) slightly because we want 
    # the AI to play its best moves, not explore weird variations.
    mcts = MCTS(model, c_puct=1.0) 
    
    print("\nWelcome to Human vs AI Tic-Tac-Toe!")
    
    # Let the human choose who goes first
    choice = input("Do you want to play as X (goes first) or O (goes second)? (x/o): ").strip().lower()
    human_player = 1 if choice == 'x' else -1
    
    # 3. The Game Loop
    while not board.is_game_over():
        if board.turn == human_player:
            print_board(board)
            move = -1
            while move not in board.legal_moves:
                try:
                    move = int(input("Enter your move (0-8): "))
                    if move not in board.legal_moves:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid number.")
            board.push(move)
        else:
            print("AI is calculating...")
            # We give the AI 500 simulations per turn. 
            # Because it is a tiny game, this will take less than 0.1 seconds.
            move_visits = mcts.search(board, simulations=500)
            
            # The AI picks the move it visited the most during its simulation
            best_move = max(move_visits, key=move_visits.get)
            board.push(best_move)
            print(f"AI plays at position {best_move}")
            
    # 4. Game Over Evaluation
    print_board(board)
    result = board.result()
    
    if result == '1/2-1/2':
        print("Result: DRAW. (The AI played perfectly!)")
    elif (result == '1-0' and human_player == 1) or (result == '0-1' and human_player == -1):
        print("Result: YOU WIN! (The AI is not perfect yet. Keep training!)")
    else:
        print("Result: AI WINS! (You blundered!)")

if __name__ == "__main__":
    play()
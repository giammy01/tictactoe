import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from tictactoe import TicTacToeBoard, encode_board
from modelttt import TicTacToeNet
import random
import os
from collections import deque

class Node:
    def __init__(self, state, prior_prob):
        self.state = state
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}

    def q_value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, c_puct=1.5):
        self.model = model
        self.c_puct = c_puct

    @torch.no_grad()
    def search(self, root_state, simulations=50):
        root = Node(root_state, prior_prob=1.0)
        
        for _ in range(simulations):
            node = root
            temp_board = root_state.copy()
            
            # FIX: Track the exact path taken to reach the leaf node
            search_path = [root] 
            
            # 1. Selection
            while len(node.children) > 0:
                best_ucb = -float('inf')
                best_move, best_child = None, None
                for move, child in node.children.items():
                    ucb = -child.q_value() + self.c_puct * child.prior_prob * (
    math.sqrt(node.visit_count) / (1 + child.visit_count))
                    if ucb > best_ucb:
                        best_ucb, best_move, best_child = ucb, move, child
                node = best_child
                temp_board.push(best_move)
                search_path.append(node) # Add node to the path
                
            # 2. Evaluation & Expansion
            if not temp_board.is_game_over():
                tensor_state = encode_board(temp_board)
                policy_logits, value = self.model(tensor_state)
                
                policy_probs = F.softmax(policy_logits, dim=1)[0].numpy()
                legal_moves = temp_board.legal_moves
                
                for move in legal_moves:
                    node.children[move] = Node(temp_board, prior_prob=policy_probs[move])
                    
                value = value.item()
            else:
                res = temp_board.result()
                if res == '1-0': value = 1.0 if temp_board.turn == 1 else -1.0
                elif res == '0-1': value = -1.0 if temp_board.turn == -1 else 1.0
                else: value = 0.0
                
            # 3. Backpropagation (FIX: Minimax value flipping)
            # Walk backward up the tree. Flip the value at each step 
            # because every step up is the opponent's turn.
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.value_sum += value
                value = -value 
                
        return {move: child.visit_count for move, child in root.children.items()}
        

def self_play(model, games=10):
    model.eval()
    training_data = []
    
    for _ in range(games):
        board = TicTacToeBoard()
        mcts = MCTS(model)
        states, policies = [], []
        
        while not board.is_game_over():
            # FIX 1: Give the AI enough budget to think ahead
            move_visits = mcts.search(board, simulations=100)
            total_visits = sum(move_visits.values())
            
            policy_target = [0.0] * 9
            for m, v in move_visits.items():
                policy_target[m] = v / total_visits
                
            states.append(board.copy())
            policies.append(policy_target)
            
            # FIX 2: Probabilistic sampling to force game variety
            moves = list(move_visits.keys())
            weights = [move_visits[m] / total_visits for m in moves]
            
            # Roll a weighted die to pick the move instead of hardcoding max()
            best_move = random.choices(moves, weights=weights, k=1)[0]
            
            board.push(best_move)
            
        res = board.result()
        if res == '1-0': val = 1.0
        elif res == '0-1': val = -1.0
        else: val = 0.0
        
        for state, policy in zip(states, policies):
            v = val if state.turn == 1 else -val
            training_data.append((state, policy, v))
            
    return training_data


def train():
    model = TicTacToeNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # FIX 1: Cure the amnesia. Load the model if it exists!
    if os.path.exists("tictactoe_model.pth"):
        model.load_state_dict(torch.load("tictactoe_model.pth"))
        print("Successfully loaded saved weights. Resuming training...")
        
    # FIX 2: Create a Replay Buffer (remembers the last 10,000 moves)
    replay_buffer = deque(maxlen=10000)
    
    generation = 1
    while True:
        print(f"Generation {generation} - Generating self-play games...")
        
        # Generate new data and add it to our historical memory
        new_data = self_play(model, games=25) 
        replay_buffer.extend(new_data)
        
        # Sample a random mix of old and new data to train on
        # This prevents the network from forgetting how to play
        train_data = random.sample(replay_buffer, min(len(replay_buffer), 1000))
        
        model.train()
        total_loss = 0
        batch_size = 32
        
        # Train on the sampled historical batch
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            state_batch = torch.cat([encode_board(s) for s, p, v in batch])
            policy_batch = torch.tensor([p for s, p, v in batch], dtype=torch.float32)
            value_batch = torch.tensor([[v] for s, p, v in batch], dtype=torch.float32)
            
            optimizer.zero_grad()
            
            pred_policy_logits, pred_value = model(state_batch)
            
            v_loss = F.mse_loss(pred_value, value_batch)
            p_loss = F.cross_entropy(pred_policy_logits, policy_batch) 
            
            loss = v_loss + p_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            
        print(f"Loss: {total_loss/len(train_data):.4f}")
        
        if generation % 5 == 0:
            torch.save(model.state_dict(), "tictactoe_model.pth")
            print("Model weights saved to disk.")
            
        generation += 1
        
if __name__ == "__main__":
    train()
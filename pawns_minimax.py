"""
Written as Task 1 of CRUx inductions. Objective is to code a minimax algorithm that can play a simple pawn-based
chess game. Algorithm uses a rudimentary self-designed heuristic algorithm, which scores a position as follows:
1. Each white pawn at the final rank returns +10
2. Each white pawn that can capture a black pawn returns +2
3. Each step away from the initial square returns +(steps away from start)/N, where N is the length of a side
4. Each black pawn follows the same scoring scheme as a white pawn, except with signs inverted
Algorithm is non ideal since the depth of the game tree search is not necessarily equal to the depth of terminal states.
Hence, the game is unsolved, and the AI is fallible.
Alpha-Beta pruning not applied, since a terminal state is not always reached, and unexplored branches could yield better
solutions.
Game tree search depth made variable to user upto a limit (10 levels), as time complexity exponentially increases with
increasing size of board.
"""

import copy

n = 5
w = "W"
b = "B"
EMPTY = " "
player = w
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rows = {}
YELLOW = '\033[33m'
GREEN = '\33[32m'
CEND = '\033[0m'
DEPTH = 6



def printboard(state):
    print()
    for i in range(0, n):
        print(YELLOW + str(i + 1) + CEND, end=" ")
        print(state[i])
    print(YELLOW + " ", [l for l in rows], CEND)
    print()


def find_pawns(state, current_player):
    bpawns, wpawns = [], []
    for i in range(0, n):
        for j in range(0, n):
            if state[i][j] == w:
                wpawns.append((i, j))
            elif state[i][j] == b:
                bpawns.append((i, j))
    if current_player == w:
        return wpawns
    else:
        return bpawns


def islegal(state, fro, to, player):
    if not exists(to):
        return False
    if player == w:
        if state[fro[0]][fro[1]] != w:
            return False
        if to[0] - fro[0] == -1:
            if fro[1] == to[1] and state[to[0]][to[1]] == EMPTY:
                return True
            elif abs(fro[1] - to[1]) == 1 and state[to[0]][to[1]] == b:
                return True
    elif player == b:
        if state[fro[0]][fro[1]] != b:
            return False
        if to[0] - fro[0] == 1:
            if fro[1] == to[1] and state[to[0]][to[1]] == EMPTY:
                return True
            elif abs(fro[1] - to[1]) == 1 and state[to[0]][to[1]] == w:
                return True


def exists(pos):
    if 0 <= pos[0] < n and 0 <= pos[1] < n:
        return True
    return False


def possiblemoves(state, current_player):
    moves = []
    if current_player == w:
        pawns = find_pawns(state, current_player)
        for pawn in pawns:
            if islegal(state, pawn, (pawn[0] - 1, pawn[1]), current_player):
                moves.append((pawn, (pawn[0] - 1, pawn[1])))
            if islegal(state, pawn, (pawn[0] - 1, pawn[1] + 1), current_player):
                moves.append((pawn, (pawn[0] - 1, pawn[1] + 1)))
            if islegal(state, pawn, (pawn[0] - 1, pawn[1] - 1), current_player):
                moves.append((pawn, (pawn[0] - 1, pawn[1] - 1)))
    elif current_player == b:
        pawns = find_pawns(state, current_player)
        for pawn in pawns:
            if islegal(state, pawn, (pawn[0] + 1, pawn[1]), current_player):
                moves.append((pawn, (pawn[0] + 1, pawn[1])))
            if islegal(state, pawn, (pawn[0] + 1, pawn[1] + 1), current_player):
                moves.append((pawn, (pawn[0] + 1, pawn[1] + 1)))
            if islegal(state, pawn, (pawn[0] + 1, pawn[1] - 1), current_player):
                moves.append((pawn, (pawn[0] + 1, pawn[1] - 1)))
    if len(moves) == 0:
        return None
    return moves


def winner(state):
    w_score, b_score = 0, 0
    for i in state[0]:
        if i == w:
            w_score += 1
    for i in state[n - 1]:
        if i == b:
            b_score += 1
    if w_score > b_score:
        return w
    elif w_score < b_score:
        return b
    else:
        return None


def new_state(state, fro, to, current_player):
    new = copy.deepcopy(state)
    if islegal(state, fro, to, current_player):
        new[fro[0]][fro[1]], new[to[0]][to[1]] = EMPTY, new[fro[0]][fro[1]]
    return new


def terminal_value(result):
    if result == w:
        return 10000
    elif result == b:
        return -10000
    else:
        return 0


def is_terminal(state, current_player):
    if possiblemoves(state, current_player) is None:
        return True
    return False


def evaluate_state(state, current_player):
    score = 0
    if is_terminal(state, current_player):
        return terminal_value(winner(state))
    else:
        positions = find_pawns(state, w)
        for pos in state[0]:
            if pos == w:
                score += 10
        for pos in positions:
            score += (1 / n) * (n - 1 - pos[0])
            if pos[0] - 1 >= 0 and 0 <= pos[1] - 1 < n and state[pos[0] - 1][pos[1] - 1] == b:
                score -= 2
            if pos[0] - 1 >= 0 and 0 <= pos[1] + 1 < n and state[pos[0] - 1][pos[1] + 1] == b:
                score -= 2
        free = True
        for pos in positions:
            for i in range(0, pos[0] + 1, 1):
                if state[i][pos[1]] is not EMPTY:
                    free = False
        if free:
            score += 4
        positions = find_pawns(state, b)
        for pos in state[-1]:
            if pos == b:
                score -= 10
        for pos in positions:
            score -= (1 / n) * (pos[0])
            if pos[0] + 1 < n and 0 <= pos[1] - 1 < n and state[pos[0] + 1][pos[1] - 1] == w:
                score += 2
            if pos[0] + 1 < n and 0 <= pos[1] + 1 < n and state[pos[0] + 1][pos[1] + 1] == w:
                score += 2
        free = True
        for pos in positions:
            for i in range(pos[0], n, 1):
                if state[i][pos[1]] is not EMPTY:
                    free = False
        if free:
            score -= 4
    return score


def ai_turn(state, current_player):
    return minimax(state, current_player)
    # return random.choice(possiblemoves(state, current_player))


def translate(fro, to):
    new_fro = (int(fro[1]) - 1, rows[fro[0].upper()])
    new_to = (int(to[1]) - 1, rows[to[0].upper()])
    return new_fro, new_to


def user_turn(state, current_player):
    fro = input("Enter square from which you would like to move your piece: ").strip()
    to = input("Enter square to which you would like to move your piece: ").strip()
    while len(fro) != 2 and len(to) != 2 and not fro[0].isalpha() and not to[0].isalpha() and not fro[1].isnumeric() \
            and not to[1].isnumeric():
        print("Invalid move(s)! Try again.")
        fro, to = user_turn(state, current_player)
    fro, to = translate(fro, to)
    while not islegal(state, fro, to, current_player):
        print("Invalid move(s)! Try again.")
        fro, to = user_turn(state, current_player)
    return fro, to


def maxval(state, current_player, current_depth):
    if current_player == w:
        not_current_player = b
    else:
        not_current_player = w
    val = -10000
    actions = possiblemoves(state, current_player)
    if is_terminal(state, current_player):
        return evaluate_state(state, current_player)
    elif current_depth == DEPTH:
        # print(evaluate_state(state, current_player))
        return evaluate_state(state, current_player)
    else:
        for action in actions:
            val = max(val, minval(new_state(state, action[0], action[1], current_player), not_current_player,
                                  current_depth + 1))
    return val


def minval(state, current_player, current_depth):
    if current_player == w:
        not_current_player = b
    else:
        not_current_player = w
    val = 10000
    actions = possiblemoves(state, current_player)
    if is_terminal(state, current_player):
        return evaluate_state(state, current_player)
    elif current_depth == DEPTH:
        # print(evaluate_state(state, current_player))
        return evaluate_state(state, current_player)
    else:
        for action in actions:
            val = min(val, maxval(new_state(state, action[0], action[1], current_player), not_current_player,
                                  current_depth + 1))
    return val


def minimax(state, ai):
    if is_terminal(state, ai):
        print("TERMINAL STATE")
        return None
    else:
        if ai == w:
            optimal_action = None
            actions = possiblemoves(state, ai)
            bestval = -10000
            for action in actions:
                value = minval(new_state(state, action[0], action[1], ai), b, 0)
                # print(value, end=" ")
                if value > bestval:
                    bestval = value
                    optimal_action = action
            # print()
            return optimal_action
        elif ai == b:
            optimal_action = None
            actions = possiblemoves(state, ai)
            bestval = 10000
            for action in actions:
                value = maxval(new_state(state, action[0], action[1], ai), w, 0)
                # print(value, end=" ")
                if value < bestval:
                    bestval = value
                    optimal_action = action
            # print()
            return optimal_action


def print_instructions():
    print("   Welcome to March of Pawns!")
    print("1. The objective of the game is to end up with as many pawns on your opponent's home rank.")
    print("2. The game consists of an NxN chessboard and N pawns for each player. White starts, as always. ")
    print("3. Pawns can either capture another pawn of the opponents with a single square diagonal movement or ")
    print("   advance 1 square (basic chess rules of a pawn). No En Passant or initial 2 square advances allowed.")
    print("4. The terminal state is when the player whose turn it has no allowed movements (all current pawns are ")
    print("   blocked by another pawn, can’t capture, or reached the other end). The winner is decided by who has more")
    print("   pawns at the opponent's end. If both have the same number, it’s a draw.")
    print("5. While entering from and to coordinates, enter the chosen coordinate as regular chess notation, with rank")
    print("   order inverted, i.e., white starts from the Nth rank, and black the first. File order stays the same.")
    print("   eg. A5, b6, c3 are all valid coordinates.")
    print("   That's all the instructions you need. Good luck!")
    print()


if __name__ == '__main__':
    print_instructions()
    n = int(input("Enter size of board as N (for NxN sized board): "))
    user = input("Would you like to play as white or black? (w/b): ").upper()
    gameover = False
    user_move = True
    ai = b
    DEPTH = (input("Enter max depth of game tree search (2-10, lower depth for higher size): "))
    while not DEPTH.isnumeric() and not 1 <= int(DEPTH) <= 10:
        DEPTH = (input("Enter max depth of game tree search (2-10, lower depth for higher size): "))
    DEPTH = int(DEPTH)
    while user != w and user != b:
        user = input("Would you like to play as white or black? (w/b): ").upper()
    if user == w:
        user_move = True
        ai = b
    elif user == b:
        user_move = False
        ai = w
    for i in range(0, n):
        rows[letters[i]] = i
    board = [[EMPTY for _ in range(n)] for _ in range(n)]
    board[n - 1] = [w for _ in range(n)]
    board[0] = [b for _ in range(n)]
    print("GAMEPLAY BEGINS")
    print()
    while not gameover:
        print()
        printboard(board)
        print()
        if user_move:
            user_move = user_turn(board, user)
            board = new_state(board, user_move[0], user_move[1], user)
            user_move = False
            if is_terminal(board, ai):
                gameover = True
                break
        if not user_move:
            print()
            print(GREEN + "AI is thinking..." + CEND)
            print()
            ai_move = ai_turn(board, ai)
            print(ai_move)
            board = new_state(board, ai_move[0], ai_move[1], ai)
            user_move = True
            if is_terminal(board, user):
                gameover = True
                break
    game_winner = winner(board)
    printboard(board)
    if game_winner == user:
        print("Well done! You have won the game")
    elif game_winner == ai:
        print('Oh no, you have lost. Better luck next time!')
    elif game_winner is None:
        print("Well played! It is a draw.")

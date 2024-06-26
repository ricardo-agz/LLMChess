from copy import deepcopy
from typing import List, Tuple, Optional
import random
import time

from chess_board import ChessBoard, Position
from pieces.chess_piece import ChessPiece, PlayerColor

from util import string_to_position, position_to_string


def minimax(
    board_state: ChessBoard,
    depth: int,
    player_color: PlayerColor,
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    cache: Optional[
        dict[str, tuple[Optional[ChessPiece], Optional[Position], int]]
    ] = None,
    start_time: time = None,
    time_limit: time = None,
    lmr_move_count: int = 100,
) -> Tuple[Optional[ChessPiece], Optional[Position], int]:
    """
    Minimax algorithm with alpha-beta pruning for the chess AI

    Args:
        board_state (ChessBoard): Current state of the chess board.
        depth (int): Depth of the search tree.
        player_color (PlayerColor): Color of the current player.
        alpha (float, optional): Alpha value for alpha-beta pruning. Defaults to -float('inf').
        beta (float, optional): Beta value for alpha-beta pruning. Defaults to float('inf').
        cache (Optional[dict[str, tuple[Optional[ChessPiece], Optional[Position], int]]], optional): A dictionary to store previously computed board evaluations. Defaults to None.
        lmr_move_count (int): how many moves to do full depth search, rest do shallower search
    Returns:
        Tuple[Optional[ChessPiece], Optional[Position], int, bool]: Best piece, best move, score of the best move, terminated due to time.
    """

    if cache is None:
        cache = {}

    board_key = hash(board_state)
    if board_key in cache and depth == cache[board_key][2]:
        # print(f'board key {board_key}')
        # print(cache[board_key])
        return cache[board_key]

    elapsed_time = time.time() - start_time
    terminate = (
        start_time is not None and time_limit is not None and elapsed_time >= time_limit
    )

    opponent_color = (
        PlayerColor.WHITE if player_color == PlayerColor.BLACK else PlayerColor.BLACK
    )
    maximizing_player = (
        player_color == PlayerColor.WHITE
    )  # white is always maximizing, black minimizing

    # base case: depth is 0 or game over
    if (
        depth == 0
        or board_state.is_checkmate(player_color)
        or board_state.is_stalemate(player_color)
        or terminate
    ):
        terminated_score = None
        evaluated_score = (
            board_state.evaluation_function() if depth == 0 else terminated_score
        )

        return None, None, evaluated_score, terminate

    best_move = None
    best_piece = None

    if maximizing_player:
        max_score = -float("inf")

        # Get all possible moves and their scores
        possible_moves = [
            (piece, move)
            for piece, moves in board_state.get_possible_moves(player_color)
            for move in moves
        ]

        # Sort the moves based on their scores
        possible_moves.sort(
            key=lambda move: move_score(move, board_state), reverse=True
        )  # descending order

        terminated = False

        # Iterate over the ordered moves
        for move_num, (piece, move) in enumerate(possible_moves):
            # create a new board and move the piece
            new_board = deepcopy(board_state)
            new_piece = deepcopy(piece)
            new_board.move_piece(new_piece, move)

            # Late Move Reductions
            reduction = 1 if move_num <= lmr_move_count else 2
            minimax_piece, minimax_move, minimax_score, terminated_lmr = minimax(
                board_state=new_board,
                depth=depth - reduction,
                player_color=opponent_color,
                alpha=alpha,
                beta=beta,
                cache=cache,
                start_time=start_time,
                time_limit=time_limit,
            )

            if minimax_score is not None and reduction == 2 and minimax_score > alpha:
                minimax_piece, minimax_move, minimax_score, terminated_deep = minimax(
                    board_state=new_board,
                    depth=depth - 1,
                    player_color=opponent_color,
                    alpha=alpha,
                    beta=beta,
                    cache=cache,
                    start_time=start_time,
                    time_limit=time_limit,
                )

            # update best move if a better score is found
            if minimax_score is not None and minimax_score > max_score:
                max_score = minimax_score
                best_move = move
                best_piece = piece

            terminated = terminated_lmr

            # Update alpha and prune if beta <= alpha only after the full depth search
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break

        max_score = None if max_score == -float("inf") else max_score

        cache[board_key] = best_piece, best_move, max_score
        return best_piece, best_move, max_score, terminated
    else:
        min_score = float("inf")

        # Get all possible moves and their scores
        possible_moves = [
            (piece, move)
            for piece, moves in board_state.get_possible_moves(player_color)
            for move in moves
        ]

        # Sort the moves based on their scores
        possible_moves.sort(
            key=lambda move: move_score(move, board_state), reverse=False
        )  # ascending order

        terminated = False

        for move_num, (piece, move) in enumerate(possible_moves):
            # create a new board and move the piece
            new_board = deepcopy(board_state)
            new_piece = deepcopy(piece)
            new_board.move_piece(new_piece, move)

            # Late Move Reductions
            reduction = 1 if move_num <= lmr_move_count else 2
            minimax_piece, minimax_move, minimax_score, terminated_lmr = minimax(
                board_state=new_board,
                depth=depth - reduction,
                player_color=opponent_color,
                alpha=alpha,
                beta=beta,
                cache=cache,
                start_time=start_time,
                time_limit=time_limit,
            )

            if minimax_score is not None and reduction == 2 and minimax_score < beta:
                minimax_piece, minimax_move, minimax_score, terminated_deep = minimax(
                    board_state=new_board,
                    depth=depth - 1,
                    player_color=opponent_color,
                    alpha=alpha,
                    beta=beta,
                    cache=cache,
                    start_time=start_time,
                    time_limit=time_limit,
                )

            # update best move if a lower score is found
            if minimax_score is not None and minimax_score < min_score:
                min_score = minimax_score
                best_move = move
                best_piece = piece

            terminated = terminated_lmr

            # update beta and prune if beta <= alpha
            beta = min(beta, min_score)
            if beta <= alpha:
                break

        min_score = None if min_score == float("inf") else min_score

        cache[board_key] = best_piece, best_move, min_score
        return best_piece, best_move, min_score, terminated


def iterative_deepening_minimax(
    board_state: ChessBoard,
    max_depth: int,
    player_color: PlayerColor,
    time_limit: int,
) -> Tuple[Optional[ChessPiece], Optional[Position], int]:
    start_time = time.time()
    maximizing_player = player_color == PlayerColor.WHITE

    depth_move_scores = []

    for current_depth in range(1, max_depth + 1):
        print(f"Depth: {current_depth}")
        piece, move, score, terminated = minimax(
            board_state=board_state,
            depth=current_depth,
            player_color=player_color,
            start_time=start_time,
            time_limit=time_limit,
        )

        if not terminated:
            print(f"Best move at depth: {current_depth}: {piece}, {move}, {score}")
            depth_move_scores.append((piece, move, score))
        else:
            print(f"Search at depth = {current_depth} was terminated")
            best_score = depth_move_scores[-1][2]
            terminated_search_best_score = score

            if (
                maximizing_player
                and terminated_search_best_score is not None
                and terminated_search_best_score > best_score
            ):
                print(
                    f"Explored move is better than best move at previous depth... {piece}, {move}, {score}"
                )
                depth_move_scores.append((piece, move, score))

            elif (
                not maximizing_player
                and terminated_search_best_score is not None
                and terminated_search_best_score < best_score
            ):
                print(
                    f"Explored move is better than best move at previous depth... {piece}, {move}, {score}"
                )
                depth_move_scores.append((piece, move, score))

        # check if time limit has been reached and break if so
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"Depth: {current_depth} - Time Limit Reached")
            break

    return depth_move_scores[-1]


def get_random_move(
    board_state: ChessBoard, color: PlayerColor
) -> Tuple[ChessPiece, Position]:
    possible_moves = []

    for piece, moves in board_state.get_possible_moves(color):
        for move in moves:
            possible_moves.append((piece, move))

    if not possible_moves:
        return None, None

    return random.choice(possible_moves)


sicilian_defense = [("Pawn", "C7", "C5"), ("Pawn", "D7", "D6")]
carokann_defense = [("Pawn", "C7", "C6"), ("Pawn", "D7", "D5")]
slav_defense = [("Pawn", "D7", "D5"), ("Pawn", "C7", "C6")]

book_moves_black = {
    ("Pawn", "E2", "E4"): {"sicilian": sicilian_defense, "caro-kann": carokann_defense},
    ("Pawn", "D2", "D4"): {
        "slav": slav_defense,
    },
}


def get_book_move_black(board_state: ChessBoard) -> Tuple[str, Position, Position]:
    moves = board_state.get_moves()
    turn = len(moves) // 2

    if len(moves) == 0 or turn > 1:
        return None, None, None

    white_opening = moves[0]

    # if white starts with a popular opening, choose the appropriate defense
    if white_opening in book_moves_black:
        random_book_opening = random.choice(book_moves_black[white_opening])
        move = book_moves_black[white_opening][random_book_opening][turn]
    # arbitrarily pick the caro kann defense
    else:
        move = carokann_defense[turn]

    piece_str, src_str, dest_str = move
    return piece_str, string_to_position(src_str), string_to_position(dest_str)


def move_score(move: Tuple[ChessPiece, Position], board_state: ChessBoard) -> int:
    piece, target_position = move
    target_piece = board_state.get_piece(target_position)

    score = 0
    see_score = 0
    # capture moves given priority based on relative value
    if target_piece is not None and target_piece.color != piece.color:
        # what should multiplier be?
        see_score = (
            static_exchange_evaluation(board_state, (piece, target_position)) * 100
        )
        score += see_score
        # print(score)
        # score += (target_piece.value - piece.value) * 50

    new_board = deepcopy(board_state)
    new_piece = deepcopy(piece)
    new_board.move_piece(new_piece, target_position)

    # check moves also given priority
    opponent_color = (
        PlayerColor.WHITE if piece.color == PlayerColor.BLACK else PlayerColor.BLACK
    )
    if new_board.is_king_in_check(opponent_color):
        score += 50

    # center control bonus
    central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    if target_position in central_squares:
        score += 10

    # moved piece mobility is rewarded
    mobility = len(new_piece.get_possible_moves(new_board))
    score += mobility

    # if piece.color == PlayerColor.BLACK:
    #     score = -score
    # score += new_board.evaluation_function()
    return score


def get_best_move(
    board_state: ChessBoard,
    color: PlayerColor,
    max_depth: int = None,
    max_time: int = None,
) -> Tuple[ChessPiece, Position]:
    time_limit = max_time  # time limit in seconds
    max_depth = max_depth

    if color == PlayerColor.BLACK:
        book_move = get_book_move_black(board_state)
        piece_str, start_pos, end_pos = book_move
        if piece_str is not None:
            piece = board_state.get_piece(start_pos)
            return piece, end_pos

    piece, move, _ = iterative_deepening_minimax(
        board_state=board_state,
        max_depth=max_depth,
        player_color=color,
        time_limit=time_limit,
    )
    # piece, move, _ = minimax(board_state, depth, True, color)
    # piece, move = get_random_move(board_state, color)

    return piece, move


def static_exchange_evaluation(
    board_state: ChessBoard, move: Tuple[ChessPiece, Position]
) -> int:
    """
    This function performs Static Exchange Evaluation (SEE) on a given move.

    Args:
        board_state (ChessBoard): The current state of the chess board.
        move (Tuple[ChessPiece, Position]): The move to be evaluated.

    Returns:
        int: The SEE score for the given move.
    """
    piece, target_position = move
    attacker_color = piece.color
    opponent_color = (
        PlayerColor.WHITE if attacker_color == PlayerColor.BLACK else PlayerColor.BLACK
    )

    # Get the target piece, and return 0 if there is no target piece
    target_piece = board_state.get_piece(target_position)
    if target_piece is None or target_piece.color == attacker_color:
        return 0

    # Initialize the attackers and the gains array
    attackers = {color: [] for color in [attacker_color, opponent_color]}
    gains = [0] * 32
    gains[0] = target_piece.value

    # Find all attacking pieces for both sides
    for row in range(8):
        for col in range(8):
            attacking_piece = board_state.get_piece((row, col))
            if attacking_piece is not None and attacking_piece.color in [
                attacker_color,
                opponent_color,
            ]:
                if target_position in attacking_piece.get_possible_moves(board_state):
                    attackers[attacking_piece.color].append(
                        (attacking_piece, attacking_piece.value)
                    )

    # Sort the attackers by the piece values
    for color in [attacker_color, opponent_color]:
        attackers[color].sort(key=lambda x: x[1])

    current_attacker_color = attacker_color
    current_depth = 1

    while True:
        # If no attackers are left for the current side, break the loop
        if not attackers[current_attacker_color]:
            break

        # Get the next attacker
        next_attacker, next_attacker_value = attackers[current_attacker_color].pop(0)

        # Add the captured value to the gains array
        gains[current_depth] = -gains[current_depth - 1] + next_attacker_value

        # Switch to the other side
        current_attacker_color = (
            opponent_color
            if current_attacker_color == attacker_color
            else attacker_color
        )
        current_depth += 1

    # Compute the SEE score
    while current_depth > 1:
        current_depth -= 1
        gains[current_depth - 1] = min(gains[current_depth - 1], -gains[current_depth])

    return gains[0]

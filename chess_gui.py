import asyncio
import time
import tkinter as tk
import enum
import threading
import traceback

from PIL import ImageTk, Image  # pip install pillow
from typing import Optional
from chess_board import ChessBoard, Position
from pieces import Pawn, Rook, Knight, Bishop, Queen, King
from pieces.chess_piece import PlayerColor
from engine import minimax, get_best_move
from make_ai_move import make_ai_move, VALID_MODELS


FORFEIT_AFTER_K_INVALID_MOVES = 25
DRAW_AFTER_K_MOVES_WITHOUT_PAWN_OR_CAPTURE = 50


class TurnOutcome(enum.Enum):
    CONTINUE = "continue"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW = "draw"
    FORFEIT = "forfeit"


class ChessGUI(tk.Tk):
    def __init__(self, board: ChessBoard):
        super().__init__()

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.start_loop, daemon=True)
        self.loop_thread.start()

        self.board = board
        self.selected_piece: Optional[Position] = None
        self.current_player = PlayerColor.WHITE
        self.white_player = "HUMAN"
        self.black_player = "HUMAN"

        self.black_ai_memory = None
        self.white_ai_memory = None

        self.title("Chess")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=640, height=640)
        self.canvas.pack(side=tk.LEFT)

        self.move_history = tk.Text(self.frame, width=30, height=40)
        self.move_history.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.pawn_capture_history = []

        self.canvas.bind("<Button-1>", self.on_tile_click)

        # Player selection menu
        self.player_menu_frame = tk.Frame(self)
        self.player_menu_frame.pack()

        self.white_player_var = tk.StringVar(value="HUMAN")
        self.black_player_var = tk.StringVar(value="HUMAN")

        tk.Label(self.player_menu_frame, text="White Player:").pack(side=tk.LEFT)
        self.white_player_menu = tk.OptionMenu(self.player_menu_frame, self.white_player_var, "HUMAN", *VALID_MODELS)
        self.white_player_menu.pack(side=tk.LEFT)

        tk.Label(self.player_menu_frame, text="Black Player:").pack(side=tk.LEFT)
        self.black_player_menu = tk.OptionMenu(self.player_menu_frame, self.black_player_var, "HUMAN", *VALID_MODELS)
        self.black_player_menu.pack(side=tk.LEFT)

        # Buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack()

        self.start_button = tk.Button(self.buttons_frame, text="Start Game", command=self.start_game)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.restart_button = tk.Button(self.buttons_frame, text="Restart Game", command=self.restart_game, state=tk.DISABLED)
        self.restart_button.grid(row=0, column=1, padx=5, pady=5)

        self.paused = False
        self.pause_button = tk.Button(self.buttons_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=2, padx=5, pady=5)

        self.board_states = [self.board.get_board_state()]
        self.current_move_index = 0
        self.previous_button = tk.Button(self.buttons_frame, text="<", command=self.previous_move, state=tk.DISABLED)
        self.previous_button.grid(row=0, column=3, padx=5, pady=5)

        self.next_button = tk.Button(self.buttons_frame, text=">", command=self.next_move, state=tk.DISABLED)
        self.next_button.grid(row=0, column=4, padx=5, pady=5)

        # Thinking label
        self.thinking_label = tk.Label(self, text="")
        self.thinking_label.pack()

        # Thought strategy text
        self.thought_strategy_text = tk.Text(self.frame, width=30, height=40, wrap=tk.WORD)
        self.thought_strategy_text.pack(side=tk.LEFT, fill=tk.BOTH)

        # Player material count
        self.material_label = tk.Label(self, text="")
        self.material_label.pack()
        white_material, black_material = self.board.get_material_count()
        self.material_label.config(text=f"White Material: {white_material}, Black Material: {black_material}")

        # Game outcome label
        self.game_outcome_label = tk.Label(self, text="")
        self.game_outcome_label.pack()

        # load piece images
        self.white_pieces = {
            "Pawn": ImageTk.PhotoImage(Image.open("images/white_pieces/pawn.png")),
            "Rook": ImageTk.PhotoImage(Image.open("images/white_pieces/rook.png")),
            "Knight": ImageTk.PhotoImage(Image.open("images/white_pieces/knight.png")),
            "Bishop": ImageTk.PhotoImage(Image.open("images/white_pieces/bishop.png")),
            "Queen": ImageTk.PhotoImage(Image.open("images/white_pieces/queen.png")),
            "King": ImageTk.PhotoImage(Image.open("images/white_pieces/king.png")),
        }
        self.black_pieces = {
            "Pawn": ImageTk.PhotoImage(Image.open("images/black_pieces/pawn.png")),
            "Rook": ImageTk.PhotoImage(Image.open("images/black_pieces/rook.png")),
            "Knight": ImageTk.PhotoImage(Image.open("images/black_pieces/knight.png")),
            "Bishop": ImageTk.PhotoImage(Image.open("images/black_pieces/bishop.png")),
            "Queen": ImageTk.PhotoImage(Image.open("images/black_pieces/queen.png")),
            "King": ImageTk.PhotoImage(Image.open("images/black_pieces/king.png")),
        }

        self.draw_board()
        self.place_pieces()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            if len(self.board_states) == 1:
                self.previous_button.config(state=tk.DISABLED)
                self.next_button.config(state=tk.DISABLED)
            else:
                self.previous_button.config(state=tk.NORMAL)
                self.next_button.config(state=tk.NORMAL)

            if self.current_move_index == 0:
                self.previous_button.config(state=tk.DISABLED)
            if self.current_move_index == len(self.board_states) - 1:
                self.next_button.config(state=tk.DISABLED)
        else:
            self.pause_button.config(text="Pause")
            self.previous_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)

            self.current_move_index = len(self.board_states) - 1
            self.current_player = PlayerColor.WHITE if self.current_move_index % 2 == 0 else PlayerColor.BLACK
            self.board.set_board_state(self.board_states[self.current_move_index])
            self.refresh_board()

    def previous_move(self):
        if self.current_move_index > 0:
            self.current_move_index -= 1
            board_state = self.board_states[self.current_move_index]
            self.board.set_board_state(board_state)
            self.refresh_board()
            self.current_player = PlayerColor.WHITE if self.current_move_index % 2 == 0 else PlayerColor.BLACK

            if self.current_move_index == 0:
                self.previous_button.config(state=tk.DISABLED)
            if self.current_move_index < len(self.board_states) - 1:
                self.next_button.config(state=tk.NORMAL)

    def next_move(self):
        if self.current_move_index < len(self.board_states) - 1:
            self.current_move_index += 1
            board_state = self.board_states[self.current_move_index]
            self.board.set_board_state(board_state)
            self.refresh_board()
            self.current_player = PlayerColor.WHITE if self.current_move_index % 2 == 0 else PlayerColor.BLACK

            if self.current_move_index == len(self.board_states) - 1:
                self.next_button.config(state=tk.DISABLED)
            if self.current_move_index > 0:
                self.previous_button.config(state=tk.NORMAL)

    def refresh_board(self):
        self.draw_board()
        self.place_pieces()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start_game(self):
        self.white_player_menu.config(state=tk.DISABLED)
        self.black_player_menu.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.restart_button.config(state=tk.NORMAL)

        asyncio.run_coroutine_threadsafe(self.start_game_coro(), self.loop)

    async def start_game_coro(self):
        self.white_player = self.white_player_var.get()
        self.black_player = self.black_player_var.get()

        if self.white_player != "HUMAN" and self.black_player != "HUMAN":
            await self.play_ai_vs_ai()
        else:
            while True:
                if not self.paused:
                    if (self.current_player == PlayerColor.WHITE and self.white_player != "HUMAN") or \
                            (self.current_player == PlayerColor.BLACK and self.black_player != "HUMAN"):
                        ai_model = self.white_player if self.current_player == PlayerColor.WHITE else self.black_player
                        turn_outcome = await self.play_ai_move(ai_model)

                        curr_player = self.current_player.name
                        await self.refresh_board_and_switch_player()

                        if turn_outcome == TurnOutcome.CHECKMATE:
                            self.game_outcome_label.config(
                                text=f"Checkmate! {curr_player} wins!")
                            break
                        elif turn_outcome == TurnOutcome.STALEMATE:
                            self.game_outcome_label.config(text="Stalemate! The game is a draw.")
                            break
                        elif turn_outcome == TurnOutcome.DRAW:
                            self.game_outcome_label.config(text="50 moves without a pawn capture. Stalemate!")
                            break
                        elif turn_outcome == TurnOutcome.FORFEIT:
                            self.game_outcome_label.config(text=f"Player {curr_player} forfeits after more than {FORFEIT_AFTER_K_INVALID_MOVES} invalid moves.")
                            break
                await asyncio.sleep(1)

    def restart_game(self):
        # Reset the chess board
        self.board = ChessBoard()

        # Reset the current player
        self.current_player = PlayerColor.WHITE

        # Reset the selected piece
        self.selected_piece = None

        # Reset the board states
        self.board_states = [self.board.get_board_state()]

        # Reset the current move index
        self.current_move_index = 0

        # Refresh the board
        self.refresh_board()

        # Clear the move history
        self.move_history.delete("1.0", tk.END)

        # Clear the pawn capture history
        self.pawn_capture_history.clear()

        # Clear the game outcome label
        self.game_outcome_label.config(text="")

        # Reset the material count
        white_material, black_material = self.board.get_material_count()
        self.material_label.config(text=f"White Material: {white_material}, Black Material: {black_material}")

        # Reset the player selection
        self.white_player_var.set("HUMAN")
        self.black_player_var.set("HUMAN")

        # Re-enable the player selection menus
        self.white_player_menu.config(state=tk.NORMAL)
        self.black_player_menu.config(state=tk.NORMAL)

        # Re-enable the start button
        self.start_button.config(state=tk.NORMAL, text="Start Game")

        # Disable the pause button
        self.paused = False
        self.pause_button.config(text="Pause")
        self.pause_button.config(state=tk.DISABLED)

        # Disable the previous and next buttons
        self.previous_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)

        # Clear the thinking label
        self.thinking_label.config(text="")
        self.thought_strategy_text.delete('1.0', tk.END)

        # Disable the restart button
        self.restart_button.config(state=tk.DISABLED)

    async def play_ai_vs_ai(self):
        while True:
            if not self.paused:
                if self.current_player == PlayerColor.WHITE:
                    ai_model = self.white_player
                else:
                    ai_model = self.black_player

                try:
                    turn_outcome = await self.play_ai_move(ai_model)
                except Exception as e:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                    break
                curr_player = self.current_player.name
                await self.refresh_board_and_switch_player()

                if turn_outcome == TurnOutcome.CHECKMATE:
                    print(f"Checkmate! {curr_player} wins!")
                    self.game_outcome_label.config(text=f"Checkmate! {curr_player} wins!")
                    break
                elif turn_outcome == TurnOutcome.STALEMATE:
                    print("Stalemate! The game is a draw.")
                    self.game_outcome_label.config(text="Stalemate! The game is a draw.")
                    break
                elif turn_outcome == TurnOutcome.DRAW:
                    print("50 moves without a pawn capture. Stalemate!")
                    self.game_outcome_label.config(text="50 moves without a pawn capture. Stalemate!")
                    break
                elif turn_outcome == TurnOutcome.FORFEIT:
                    self.game_outcome_label.config(
                        text=f"Player {curr_player} forfeits after more than {FORFEIT_AFTER_K_INVALID_MOVES} invalid moves.")
                    break

            await asyncio.sleep(1)

    async def play_ai_move(self, ai_model) -> TurnOutcome:
        prev_invalid_moves = []
        while True:
            if len(prev_invalid_moves) >= FORFEIT_AFTER_K_INVALID_MOVES:
                print(f"Player {self.current_player.name} forfeits after {FORFEIT_AFTER_K_INVALID_MOVES} invalid moves.")
                return TurnOutcome.FORFEIT

            if not self.paused:
                self.thinking_label.config(text=f"{ai_model} is thinking...")
                self.update()

                memory = self.white_ai_memory if self.current_player == PlayerColor.WHITE else self.black_ai_memory

                print(f"Getting {self.current_player.name} move using {ai_model}...")
                ai_move = await make_ai_move(
                    board=self.board,
                    current_player=self.current_player,
                    ai_model=ai_model,
                    prev_invalid_moves=prev_invalid_moves,
                    memory=memory,
                    move_history=self.move_history.get("1.0", tk.END),
                    last_k_move_history=5
                )
                print(f"{ai_model}'s move: {ai_move}")
                if self.paused:
                    continue
            else:
                continue

            self.thinking_label.config(text="")
            self.update()

            piece_name = ai_move['piece']
            source_str = ai_move['source']
            destination_str = ai_move['destination']
            memory = ai_move['memory']
            self.thought_strategy_text.delete('1.0', tk.END)
            self.thought_strategy_text.insert(tk.END, f"{ai_model.upper()}:\n{memory}")

            try:
                source = self.board.get_position_tuple(source_str)
                destination = self.board.get_position_tuple(destination_str)
            except Exception as e:
                source = None
                destination = None

            pieces = self.board.get_pieces(self.current_player)
            piece = next((p for p in pieces if p.__class__.__name__ == piece_name and p.position == source), None) if source else None

            if piece and destination:
                valid_move, was_pawn_or_capture = self.board.move_piece(piece, destination)
                if valid_move:
                    self.board_states.append(self.board.get_board_state())
                    self.current_move_index = len(self.board_states) - 1

                    if self.current_player == PlayerColor.WHITE:
                        self.white_ai_memory = memory
                    else:
                        self.black_ai_memory = memory

                    # Clear the previous invalid moves list
                    prev_invalid_moves.clear()

                    # Store the current player's color before switching
                    current_player_color = self.current_player

                    # Update move history using the stored player color
                    move_text = f"{current_player_color.name.capitalize()}: {piece_name} - {self.board.get_position_string(source)} -> {self.board.get_position_string(destination)}\n"
                    self.move_history.insert(tk.END, move_text)
                    self.pawn_capture_history.append(was_pawn_or_capture)

                    k_moves = DRAW_AFTER_K_MOVES_WITHOUT_PAWN_OR_CAPTURE
                    if len(self.pawn_capture_history) >= k_moves and all([not x for x in self.pawn_capture_history[-k_moves:]]):
                        print(f"{k_moves} moves without a pawn capture. Stalemate!")
                        return TurnOutcome.DRAW

                    # Check for checkmate and stalemate using the opposite player
                    opposite_player = PlayerColor.BLACK if current_player_color == PlayerColor.WHITE else PlayerColor.WHITE
                    if self.board.is_checkmate(opposite_player):
                        print(f"Checkmate! {current_player_color.name} wins!")
                        return TurnOutcome.CHECKMATE
                    elif self.board.is_stalemate(opposite_player):
                        print("Stalemate! The game is a draw.")
                        return TurnOutcome.STALEMATE
                    else:
                        print("Continuing the game...")
                        return TurnOutcome.CONTINUE
                else:
                    print("Invalid move returned by the AI.", ai_move)
                    if source and destination:
                        if all([x in range(8) for x in [source[0], source[1], destination[0], destination[1]]]):
                            self.highlight_square(source[0], source[1], "red")
                            self.highlight_square(destination[0], destination[1], "red")

                    # Add the invalid move to the list
                    prev_invalid_moves.append(ai_move)
                    continue
            else:
                print("Invalid move returned by the AI.")
                if source and destination:
                    if all([x in range(8) for x in [source[0], source[1], destination[0], destination[1]]]):
                        self.highlight_square(source[0], source[1], "red")
                        self.highlight_square(destination[0], destination[1], "red")

                # Add the invalid move to the list
                prev_invalid_moves.append(ai_move)
                continue

    async def switch_player(self):
        if self.current_player == PlayerColor.WHITE:
            self.current_player = PlayerColor.BLACK
        else:
            self.current_player = PlayerColor.WHITE

        white_material, black_material = self.board.get_material_count()
        self.material_label.config(text=f"White Material: {white_material}, Black Material: {black_material}")

    async def refresh_board_and_switch_player(self):
        self.refresh_board()
        await self.switch_player()

    def place_image(self, row, col):
        piece = self.board.get_piece((row, col))
        if piece:
            if piece.color == PlayerColor.WHITE:
                piece_image = self.white_pieces[piece.__class__.__name__]
            else:
                piece_image = self.black_pieces[piece.__class__.__name__]
            self.canvas.create_image(
                col * 80 + 40, row * 80 + 40, image=piece_image, tags=("piece", f"{row},{col}")
            )
            self.canvas.create_image(
                col * 80 + 40,
                row * 80 + 40,
                image=piece_image,
                tags=("piece", f"{row},{col}")
            )

    def highlight_square(self, row, col, color: str):
        x1, y1 = col * 80, row * 80
        x2, y2 = x1 + 80, y1 + 80
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=color, tags="highlight", stipple="gray12"
        )
        self.place_image(row, col)
        if color == "red":
            self.after(1000, self.remove_square_highlight, row, col)

    def remove_square_highlight(self, row, col):
        x1, y1 = col * 80, row * 80
        x2, y2 = x1 + 80, y1 + 80
        color = "lemon chiffon" if (row + col) % 2 == 0 else "sienna4"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        self.place_image(row, col)

    def draw_board(self):
        self.canvas.delete("highlight")
        self.canvas.delete("piece")
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 80, row * 80
                x2, y2 = x1 + 80, y1 + 80
                color = "lemon chiffon" if (row + col) % 2 == 0 else "sienna4"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def on_tile_click(self, event):
        col, row = event.x // 80, event.y // 80
        clicked_position = (row, col)
        clicked_piece = self.board.get_piece(clicked_position)

        if (self.current_player == PlayerColor.WHITE and self.white_player == "HUMAN") or \
                (self.current_player == PlayerColor.BLACK and self.black_player == "HUMAN"):
            if not self.selected_piece:
                if clicked_piece and clicked_piece.color == self.current_player:
                    self.selected_piece = clicked_piece
                    self.highlight_square(row, col, "blue")
            else:
                if clicked_piece and clicked_piece.color == self.selected_piece.color:
                    # Deselect the previously selected piece and select the new piece
                    self.remove_square_highlight(
                        self.selected_piece.position[0], self.selected_piece.position[1]
                    )
                    self.selected_piece = clicked_piece
                    self.highlight_square(row, col, "blue")
                else:
                    valid_move, was_pawn_or_capture = self.board.move_piece(
                        self.selected_piece, clicked_position
                    )
                    if valid_move:
                        # add to board states
                        self.board_states.append(self.board.get_board_state())

                        self.remove_square_highlight(
                            self.selected_piece.position[0],
                            self.selected_piece.position[1],
                        )
                        self.selected_piece = None
                        self.refresh_board()
                        asyncio.run_coroutine_threadsafe(self.switch_player(), self.loop)

                        piece_name = self.board.moves[-1][0]
                        source = self.board.get_position_string(self.board.moves[-1][1])
                        destination = self.board.get_position_string(self.board.moves[-1][2])

                        # Update move history
                        move_text = f"{self.current_player.name.capitalize()}: {piece_name} - {source} -> {destination}\n"
                        self.move_history.insert(tk.END, move_text)
                        self.pawn_capture_history.append(was_pawn_or_capture)
                    else:
                        # Invalid move
                        self.highlight_square(row, col, "red")

    def place_pieces(self):
        for row in range(8):
            for col in range(8):
                self.place_image(row, col)

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    chess_board = ChessBoard()
    gui = ChessGUI(chess_board)
    gui.run()

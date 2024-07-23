import os
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from chess_board import ChessBoard
from pieces import PlayerColor
from utils.llm_output_parser import (
    PydanticOutputParser,
    PydanticYAMLOutputParser,
    OutputFixingParser,
)


load_dotenv()

client = AsyncOpenAI(
    base_url="https://router.neutrinoapp.com/api/engines",
    api_key=os.getenv("NEUTRINO_API_KEY"),
)


VALID_MODELS = [
    "llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "claude-3.5-sonnet",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-2",
    "claude-instant-1",
    "llama-3-70b-instruct",
    "llama-3-8b-instruct",
    "command-r-plus",
    "command-r",
    "wizardlm-2-8x22b",
    "snowflake-arctic-instruct",
    "deepseek-chat",
    "dbrx-instruct",
    "mistral-large",
    "mistral-small",
    "mixtral-8x22b-instruct",
    "mixtral-8x7b-instruct",
    "mistral-7b-instruct",
]

VISION_MODELS = [
    "gpt-4o",
    "claude-3.5-sonnet",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
]


user_prompt = """
You are an AI system designed to play chess at a high level. Your task is to analyze the current state of the game 
provided below and select the best move from the list of valid options.

The current board position is shown here, with uppercase letters representing White pieces and lowercase 
representing Black:

<board>
{{BOARD_STATE}}
</board>

In this game, you are playing as the {{CURR_PLAYER}} pieces.

These are the past {{NUM_MOVE_HISTORY}} moves that have been played in the game:
<move_history>
{{MOVE_HISTORY}}
</move_history>

Based on the current position, you have the following valid moves to choose from:

<valid_moves>
{{VALID_MOVES}}
</valid_moves>

Below is your ongoing memory from previous moves. It allows you to remember your strategy and plan for the next move.
You may use this memory in your decision-making process.
<memory>
{{MEMORY}}
</memory>

Before deciding on a move, take some time to thoroughly analyze the position in a <scratchpad>. Consider important 
factors such as:
- Material balance: Which side has more valuable pieces?
- Piece activity: Which pieces are well-placed and controlling key squares?
- King safety: Are either kings exposed or vulnerable to attack?  
- Pawn structure: Are there any weak pawns or pawn chains that can be exploited?
- Tactics: Are there any immediate tactical threats or opportunities?

Carefully think through the pros and cons of the most promising candidate moves from the valid options provided. 
Aim to find the move that objectively gives {{CURR_PLAYER}} the biggest advantage going forward.

After you've thoroughly analyzed the position, select the move you think is objectively best. Do not choose any move 
that is not on the valid moves list.

Output your selected move in valid JSON format inside <move> tags. The JSON object should have keys for the piece 
moved (e.g. "Bishop", "Pawn", "Rook"), the starting square, and the destination square (e.g. "e4", "a6"). You are also 
allowed to write down one or two lines of 'memory' so you may remember your strategy for the next move.

For example:

<move>
```
{
  "piece": "Knight",
  "source": "g8",
  "destination": "f6",
  "memory": "I'm aiming to control the center and develop my pieces.",
}
```
</move>

The JSON object inside the <move> tags should strictly adhere to the format instructions outlined below:
{{FORMAT_INSTRUCTIONS}}

Remember, your goal is to select the objectively strongest move based on your analysis, not just the first decent 
option you see. Take your time and think it through carefully. I'm looking forward to seeing what move you come up 
with for this challenging position!

Here again, is your ongoing memory from previous moves. You may use this memory in your decision-making process.
{{MEMORY}}

Again, remember to consider factors such as:
- Are there any underdeveloped pieces that you can improve?
- Are any of your pieces under attack?
- Are there any pieces you can capture? Is the piece you're capturing defended?
- Are you able to put your opponent in check or checkmate?
- Can you make any trades where you come out ahead in material? Do not make any trades that are not favorable for you.
- What is the stage of the game? Are you in the opening, middlegame, or endgame?
    - In the opening, you should focus on developing your pieces and controlling the center.
    - In the middlegame, you should look for tactical opportunities and ways to improve your position.
    - In the endgame, you should aim to promote your pawns and checkmate your opponent's king.
- Is your intended destination square safe? Will your piece be vulnerable to attack after moving there?

Again, this is the state of the board:
<board>
{{BOARD_STATE}}
</board>

You are playing as the {{CURR_PLAYER}} pieces. Good luck!

{{PREV_INVALID_MOVES}}
Here again are the valid moves you can choose from:
<valid_moves>
{{VALID_MOVES}}
</valid_moves>

{{BOARD_IMAGE}}
"""


class Move(BaseModel):
    """
    Evaluation Class
    """

    piece: str = Field(description="Piece moved, e.g. 'Knight', 'Pawn', 'Rook'")
    source: str = Field(description="Starting square, e.g. 'e4', 'a6'")
    destination: str = Field(description="Destination square, e.g. 'e4', 'a6'")
    memory: str = Field(description="Memory for the next move")

    def to_dict(self) -> dict:
        return self.model_dump()


parser = PydanticOutputParser(pydantic_object=Move)
auto_fixing_parser = OutputFixingParser(
    parser=parser, model="gpt-3.5-turbo", max_retries=3
)


async def make_ai_move(
    board: ChessBoard,
    current_player: PlayerColor,
    ai_model: str = "gpt-4o",
    prev_invalid_moves: list[dict] = None,
    memory: str = None,
    move_history: str = None,
    last_k_move_history: int = 5,
    board_image: str = None,
) -> dict:
    """
    Function to make AI move
    :param board: ChessBoard: Chess board
    :param current_player: PlayerColor: Current player
    :param ai_model: str: AI model
    :param prev_invalid_moves: list[dict]: Previous invalid moves
    :param memory: str: Memory
    :param move_history: str: Move history
    :param last_k_move_history: int: Last k move history
    :param board_image: str: Board image in base64 format
    :return: dict: Move
    """
    board_state = str(board)
    valid_moves = board.get_valid_moves_str(current_player)
    curr_player = "UPPER" if current_player == PlayerColor.WHITE else "lower"
    format_instructions = parser.get_format_instructions()

    total_invalid_moves = len(prev_invalid_moves) if prev_invalid_moves else 0

    # Convert dictionaries to tuples of key-value pairs
    prev_invalid_moves_tuples = (
        [tuple(move.items()) for move in prev_invalid_moves]
        if prev_invalid_moves
        else []
    )
    prev_invalid_moves = (
        [dict(move_tuple) for move_tuple in set(prev_invalid_moves_tuples)]
        if prev_invalid_moves_tuples
        else []
    )
    prev_invalid_moves_str = ""

    last_k_invalid_moves = 3
    if len(prev_invalid_moves) > last_k_invalid_moves:
        prev_invalid_moves = prev_invalid_moves[-last_k_invalid_moves:]

    if prev_invalid_moves:
        prev_invalid_moves_str = "\n".join(
            [
                f"  - {move['piece']}: {move['source']} to {move['destination']}"
                for move in prev_invalid_moves
            ]
        )
        attempt_count = len(prev_invalid_moves)
        prev_invalid_moves_str = f"""This is attempt {attempt_count}. Previously, you tried the following moves, which 
were INVALID and NOT on the list of valid moves:
<prev_invalid_moves>
{prev_invalid_moves_str}
</prev_invalid_moves>

DO NOT repeat any of these moves as they are INVALID. You may ONLY choose from the list of valid moves provided.
"""

    if move_history:
        move_history = move_history.split("\n")
        if len(move_history) > last_k_move_history:
            move_history = move_history[-last_k_move_history:]
        move_history = "\n".join(move_history)

    memory = memory if total_invalid_moves < 5 else ""
    is_vision_model = ai_model in VISION_MODELS
    board_image_str = (
        "You will also be provided with an image of the current state of the board."
        if is_vision_model
        else ""
    )

    content_str = (
        user_prompt.replace("{{BOARD_STATE}}", board_state)
        .replace("{{VALID_MOVES}}", valid_moves)
        .replace("{{CURR_PLAYER}}", curr_player)
        .replace("{{FORMAT_INSTRUCTIONS}}", format_instructions)
        .replace("{{PREV_INVALID_MOVES}}", prev_invalid_moves_str)
        .replace("{{MEMORY}}", memory or "")
        .replace("{{MOVE_HISTORY}}", move_history or "")
        .replace("{{NUM_MOVE_HISTORY}}", str(last_k_move_history))
        .replace("{{BOARD_IMAGE}}", board_image_str)
    )

    user_content = [
        {
            "type": "text",
            "text": content_str
        },
    ]
    if is_vision_model:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{board_image}"
                }
            }
        )

    prompt_messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]
    temperature = 1

    response = await client.chat.completions.create(
        model=ai_model,
        messages=prompt_messages,
        temperature=temperature,
    )
    response_text = response.choices[0].message.content
    move_json = auto_fixing_parser.parse(response_text)

    return move_json.to_dict()

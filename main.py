"""
Multi-Agent Painter-Critic Drawing System
==========================================
Two AG2 agents collaborate iteratively to produce a digital drawing:
  - Painter: draws on a 200x200 canvas using registered tools
  - Critic: visually evaluates the canvas and provides actionable feedback

Uses AG2's initiate_chat mechanism with registered tools and reply hooks.
"""

import os
import sys
import base64
from typing import Annotated

from autogen import ConversableAgent, register_function
from openai import OpenAI
from dotenv import load_dotenv

from canvas import Canvas

# Load environment variables from .env file
load_dotenv()

# ============================================================
# Configuration
# ============================================================
NUM_ROUNDS = 10

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.environ.get("OPENAI_API_KEY")

config_list = [
    {
        "model": MODEL_NAME,
        "base_url": API_BASE_URL,
        "api_key": API_KEY,
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "cache_seed": None,  # Disable response caching
}

# ============================================================
# Reference Image Setup
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

reference_image_path = input(
    "Enter the path to a reference image (press Enter for default 'dukenukem.jpg'): "
).strip()
if not reference_image_path:
    reference_image_path = os.path.join(script_dir, "dukenukem.jpg")

if not os.path.exists(reference_image_path):
    print(f"Error: Reference image not found at {reference_image_path}")
    sys.exit(1)

# Encode reference image as base64 data URI
with open(reference_image_path, "rb") as f:
    ref_bytes = f.read()
ext = os.path.splitext(reference_image_path)[1].lower().lstrip(".")
mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
ref_mime = mime_map.get(ext, "image/png")
reference_base64 = f"data:{ref_mime};base64,{base64.b64encode(ref_bytes).decode('utf-8')}"

# ============================================================
# Canvas & Output Setup
# ============================================================
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

canvas = Canvas(200, 200)

round_tracker = {"current": 0, "target": NUM_ROUNDS}

# Separate OpenAI client for the Critic's multimodal evaluation calls
eval_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ============================================================
# Drawing Tool Wrappers
# ============================================================
def tool_draw_rectangle(
    x: Annotated[int, "Left edge X coordinate (0-199)"],
    y: Annotated[int, "Top edge Y coordinate (0-199)"],
    width: Annotated[int, "Width of the rectangle in pixels"],
    height: Annotated[int, "Height of the rectangle in pixels"],
    r: Annotated[int, "Red color component (0-255)"],
    g: Annotated[int, "Green color component (0-255)"],
    b: Annotated[int, "Blue color component (0-255)"],
) -> str:
    """Draw a filled rectangle. Best for backgrounds and large color blocks."""
    return canvas.draw_rectangle(x, y, width, height, r, g, b)


def tool_draw_line(
    x1: Annotated[int, "Start X coordinate (0-199)"],
    y1: Annotated[int, "Start Y coordinate (0-199)"],
    x2: Annotated[int, "End X coordinate (0-199)"],
    y2: Annotated[int, "End Y coordinate (0-199)"],
    r: Annotated[int, "Red color component (0-255)"],
    g: Annotated[int, "Green color component (0-255)"],
    b: Annotated[int, "Blue color component (0-255)"],
    thickness: Annotated[int, "Line thickness in pixels (default 2)"] = 2,
) -> str:
    """Draw a line between two points. Best for outlines, edges, and details."""
    return canvas.draw_line(x1, y1, x2, y2, r, g, b, thickness)


def tool_draw_circle(
    cx: Annotated[int, "Center X coordinate (0-199)"],
    cy: Annotated[int, "Center Y coordinate (0-199)"],
    radius: Annotated[int, "Radius in pixels"],
    r: Annotated[int, "Red color component (0-255)"],
    g: Annotated[int, "Green color component (0-255)"],
    b: Annotated[int, "Blue color component (0-255)"],
    fill: Annotated[bool, "True for filled circle, False for outline only"] = True,
) -> str:
    """Draw a circle on the canvas. Best for rounded shapes like eyes, heads."""
    return canvas.draw_circle(cx, cy, radius, r, g, b, fill)


# ============================================================
# Agent Definitions
# ============================================================
PAINTER_SYSTEM_MSG = """\
You are a Painter agent. Your task is to recreate a reference image on a 200×200 pixel canvas.

You have three drawing tools:
  1. tool_draw_rectangle(x, y, width, height, r, g, b)
  2. tool_draw_line(x1, y1, x2, y2, r, g, b, thickness)
  3. tool_draw_circle(cx, cy, radius, r, g, b, fill)

RULES:
• Canvas coordinates: (0,0) top-left → (199,199) bottom-right.
• Make MULTIPLE tool calls per turn (at least 5–10) for visible progress.
• Round 1: lay down backgrounds and major shapes.
• Later rounds: refine details based on the Critic's feedback.
• After your tool calls, send a SHORT text summary of what you drew.
• Pay close attention to the reference image colors and composition.
"""

CRITIC_SYSTEM_MSG = """\
You are an Art Critic agent evaluating a 200×200 pixel drawing against a reference image.

For each evaluation provide:
1. **Strengths** – specific elements that match the reference well.
2. **Weaknesses** – specific areas that differ from the reference.
3. **Actionable next steps** – concrete drawing instructions:
   • Specify coordinates, sizes, RGB colors, and which tool to use.
   • Prioritize the largest visual discrepancies first.
   • Example: "Use tool_draw_rectangle at (50,30) size 40×20 RGB(200,150,100) for the face area."

Be concise but specific. Your feedback directly guides the Painter.
"""

def _check_termination(msg):
    """Check for TERMINATE in message content (handles both str and list)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return "TERMINATE" in content
    if isinstance(content, list):
        return any(
            "TERMINATE" in (p.get("text", "") if isinstance(p, dict) else str(p))
            for p in content
        )
    return False


painter = ConversableAgent(
    name="Painter",
    system_message=PAINTER_SYSTEM_MSG,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=_check_termination,
    max_consecutive_auto_reply=100,
)

critic = ConversableAgent(
    name="Critic",
    system_message=CRITIC_SYSTEM_MSG,
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=100,
)

# ============================================================
# Register Drawing Tools (Painter proposes, Critic executes)
# ============================================================
register_function(
    tool_draw_rectangle,
    caller=painter,
    executor=critic,
    name="tool_draw_rectangle",
    description="Draw a filled rectangle on the canvas.",
)
register_function(
    tool_draw_line,
    caller=painter,
    executor=critic,
    name="tool_draw_line",
    description="Draw a line between two points with configurable thickness.",
)
register_function(
    tool_draw_circle,
    caller=painter,
    executor=critic,
    name="tool_draw_circle",
    description="Draw a circle (filled or outline) on the canvas.",
)


# ============================================================
# Hook: inject canvas + reference images for the Painter
# ============================================================
def inject_images_for_painter(messages):
    """Before the Painter's LLM call, append the current canvas and
    reference images to the last text message so the model can 'see'
    what it is working on."""
    if not messages:
        return messages

    last_msg = messages[-1]

    # Skip tool-result and tool-call messages — no image injection needed
    if last_msg.get("role") == "tool":
        return messages
    if last_msg.get("tool_calls"):
        return messages

    content = last_msg.get("content", "") or ""

    # Don't modify messages containing TERMINATE — let termination check work
    if isinstance(content, str) and "TERMINATE" in content:
        return messages
    modified = dict(last_msg)
    modified["content"] = [
        {"type": "text", "text": str(content)},
        {"type": "text", "text": "[CURRENT CANVAS]"},
        {"type": "image_url", "image_url": {"url": canvas.to_base64()}},
        {"type": "text", "text": "[REFERENCE IMAGE — recreate this]"},
        {"type": "image_url", "image_url": {"url": reference_base64}},
    ]
    # Return a new list so the stored _oai_messages stay untouched
    return list(messages[:-1]) + [modified]


painter.register_hook(
    hookable_method="process_all_messages_before_reply",
    hook=inject_images_for_painter,
)


# ============================================================
# Custom Reply: Critic evaluates the canvas with vision
# ============================================================
def critic_evaluation_reply(recipient, messages, sender, config):
    """Registered as the highest-priority reply function on the Critic.

    • For tool-call messages → fall through to the default tool-executor.
    • For text messages (Painter finished drawing) → make a multimodal
      LLM call with the canvas + reference images and return the
      evaluation.
    """
    if not messages:
        return False, None

    last_msg = messages[-1]

    # Let AG2's built-in tool-execution handler deal with tool_calls
    if last_msg.get("tool_calls"):
        return False, None
    if last_msg.get("role") == "tool":
        return False, None

    if round_tracker.get("terminate_next"):
        raise StopIteration("Target rounds reached.")

    # ---- Text message from Painter: evaluate the drawing ----
    round_tracker["current"] += 1
    current_round = round_tracker["current"]

    # Only save the canvas for rounds 1, 5, and 10
    save_path = os.path.join(output_dir, f"round_{current_round:02d}.png")
    saved_msg = ""
    if current_round in [1, 5, 10]:
        canvas.save(save_path)
        saved_msg = f"  Saved → {save_path}\n"
    
    stats = canvas.get_pixel_count()
    print(
        f"\n{'=' * 60}\n"
        f"  Round {current_round}/{round_tracker['target']}  |  "
        f"Pixels drawn: {stats['drawn_pixels']}  |  "
        f"Coverage: {stats['coverage_pct']}%\n"
        f"{saved_msg}"
        f"{'=' * 60}"
    )

    # Build a clean multimodal prompt for the Critic
    painter_note = last_msg.get("content", "") or "Drawing completed."
    eval_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_MSG},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Round {current_round} of {round_tracker['target']}.\n\n"
                        f"Painter's note: {painter_note}\n\n"
                        "Compare the CURRENT CANVAS (first image) with the "
                        "REFERENCE IMAGE (second image) and provide your evaluation."
                    ),
                },
                {"type": "image_url", "image_url": {"url": canvas.to_base64()}},
                {"type": "image_url", "image_url": {"url": reference_base64}},
            ],
        },
    ]

    # Call the LLM with vision
    try:
        response = eval_client.chat.completions.create(
            model=MODEL_NAME,
            messages=eval_messages,
            max_tokens=1500,
            temperature=0.7,
        )
        evaluation = response.choices[0].message.content
    except Exception as exc:
        evaluation = (
            f"[Evaluation error: {exc}] "
            "Please continue improving the drawing based on the reference."
        )

    # Append TERMINATE on the final round so the conversation ends
    if current_round >= round_tracker["target"]:
        evaluation += "\n\nTERMINATE"
        round_tracker["terminate_next"] = True

    return True, evaluation


critic.register_reply(
    trigger=ConversableAgent,
    reply_func=critic_evaluation_reply,
    position=0,  # Highest priority — checked before default handlers
)


# ============================================================
# Run the Conversation
# ============================================================
def main():
    print(
        f"\n{'=' * 60}\n"
        f"  Painter–Critic Multi-Agent Drawing System\n"
        f"  Reference : {reference_image_path}\n"
        f"  Canvas    : 200 × 200\n"
        f"  Rounds    : {NUM_ROUNDS}\n"
        f"  Model     : {MODEL_NAME}\n"
        f"{'=' * 60}\n"
    )

    initial_message = (
        "Please draw the reference image on the 200×200 canvas. "
        "Start with the major shapes and background colors. "
        "Use at least 5–10 tool calls this turn for visible progress."
    )

    try:
        critic.initiate_chat(
            painter,
            message=initial_message,
            max_turns=500,  # Large ceiling — actual stop is via TERMINATE
        )
    except StopIteration:
        pass

    # ---- Save conversation log ----
    log_path = os.path.join(script_dir, "conversation_log.txt")
    
    # Extract chat history
    chat_history = critic.chat_messages.get(painter, [])
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PAINTER–CRITIC CONVERSATION LOG\n")
        f.write(f"Reference image : {reference_image_path}\n")
        f.write(f"Rounds completed: {round_tracker['current']}\n")
        f.write(f"Model           : {MODEL_NAME}\n")
        f.write("=" * 80 + "\n\n")

        for idx, msg in enumerate(chat_history):
            name = msg.get("name", msg.get("role", "unknown"))
            content = msg.get("content", "")

            # Summarise tool calls compactly
            if msg.get("tool_calls"):
                lines = ["[TOOL CALLS]"]
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(
                        f"  → {fn.get('name', '?')}({fn.get('arguments', '')})"
                    )
                content = "\n".join(lines)

            # Flatten multimodal content lists to text-only
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if not content or not content.strip():
                continue

            # Truncate overly long entries (e.g. stray base64)
            content = str(content)
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"

            f.write(f"--- Message {idx + 1} [{name}] ---\n")
            f.write(f"{content}\n\n")

    print(f"\nConversation log saved → {log_path}")
    print(f"Round images saved in  → {output_dir}")
    print(f"Rounds completed       : {round_tracker['current']}")


if __name__ == "__main__":
    main()

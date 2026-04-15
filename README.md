# Multi-Agent Painter-Critic Drawing System

A multi-agent system built with the [AG2 framework](https://docs.ag2.ai/) where two AI agents collaborate iteratively to produce a digital drawing on a 200×200 pixel canvas.

## Drawing Subject

The system accepts a **user-provided reference image** and attempts to recreate it as pixel art. The default reference is `dukenukem.jpg` — a pixel-art style Duke Nukem face featuring bold colors, sunglasses, blonde hair, and a grinning expression.

## Architecture & Design Decisions

### Agent Pattern: Two-Agent Chat with Tool Calling

The system uses AG2's **two-agent conversation** pattern via `initiate_chat()`:

- **Painter Agent** (ConversableAgent with LLM) — proposes drawing tool calls
- **Critic Agent** (ConversableAgent with LLM) — executes tools and visually evaluates results

This pattern was chosen because:
1. It maps naturally to the assignment's round structure (Painter draws → Critic evaluates)
2. AG2's built-in tool execution handles the Painter→Critic tool flow automatically
3. The conversation terminates cleanly via AG2's `is_termination_msg` mechanism

### Multimodal Vision

Both agents receive visual context through the OpenAI-compatible vision API format:
- **Painter**: A `process_all_messages_before_reply` hook injects the current canvas state and reference image as base64-encoded data URIs into text messages before LLM calls
- **Critic**: A custom `register_reply` function makes a separate multimodal LLM call with both images for visual evaluation

### Drawing Tools (3 tools)

| Tool | Purpose | Why |
|------|---------|-----|
| `tool_draw_rectangle` | Fill rectangular regions with solid color | Most efficient for backgrounds and large areas — draws many pixels at once |
| `tool_draw_line` | Draw lines with configurable thickness (Bresenham's algorithm) | Essential for outlines, edges, borders, and structural elements |
| `tool_draw_circle` | Draw filled or outlined circles | Needed for curved features like eyes, head shapes, rounded details |

Tools are registered using `register_function(caller=painter, executor=critic)` — the Painter's LLM proposes calls and the Critic auto-executes them on the shared canvas.

### Round Structure

Each round consists of:
1. **Painter draws**: Makes multiple tool calls (5–10+) to draw on the canvas
2. **Critic evaluates**: Receives both the canvas and reference image visually, provides structured feedback with specific coordinates, colors, and tool suggestions

The conversation runs for a configurable number of rounds (default: 10), controlled by a round counter and `TERMINATE` message.

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Run

We strongly recommend using a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Then run the system:

```bash
python main.py
```

The system will prompt you for a reference image path. Press Enter to use the default `dukenukem.jpg`.

## Output

- `output/round_01.png`, `round_05.png`, `round_10.png` — canvas state only at keys rounds 1, 5, and 10.
- `conversation_log.txt` — full text log of the Painter↔Critic dialogue

## Configuration

Edit the constants at the top of `main.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ROUNDS` | `10` | Number of drawing rounds |
| `MODEL_NAME` | `openai/gpt-4.1-mini` | LLM model to use |
| `API_BASE_URL` | AWS proxy URL | OpenRouter-compatible API endpoint |

## Model

Uses `openai/gpt-4.1-mini` via the provided AWS proxy (no API key required). This model supports both tool calling and vision/multimodal inputs.

"""
Canvas module for the Painter-Critic multi-agent drawing system.

Provides a 200x200 pixel canvas with drawing tools that the Painter agent
can call to create and modify artwork iteratively.
"""

import base64
import io
import math

import numpy as np
from PIL import Image


class Canvas:
    """A 200x200 RGB digital canvas for pixel-art drawing."""

    def __init__(self, width: int = 200, height: int = 200):
        self.width = width
        self.height = height
        # Initialize canvas as white (255, 255, 255)
        self.pixels = np.full((height, width, 3), 255, dtype=np.uint8)

    def _clamp(self, val: int, lo: int, hi: int) -> int:
        """Clamp a value to [lo, hi]."""
        return max(lo, min(val, hi))

    def _clamp_color(self, r: int, g: int, b: int) -> tuple:
        """Clamp RGB values to [0, 255]."""
        return (
            self._clamp(r, 0, 255),
            self._clamp(g, 0, 255),
            self._clamp(b, 0, 255),
        )

    def clear(self, r: int = 255, g: int = 255, b: int = 255) -> str:
        """Reset the entire canvas to a solid color.

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).

        Returns:
            Confirmation message.
        """
        r, g, b = self._clamp_color(r, g, b)
        self.pixels[:, :] = [r, g, b]
        return f"Canvas cleared to RGB({r}, {g}, {b})."

    def draw_rectangle(
        self, x: int, y: int, width: int, height: int, r: int, g: int, b: int
    ) -> str:
        """Draw a filled rectangle on the canvas.

        This is the most efficient tool for covering large areas like backgrounds,
        color blocks, and rectangular features.

        Args:
            x: Left edge X coordinate (0-199).
            y: Top edge Y coordinate (0-199).
            width: Width of the rectangle in pixels (minimum 1).
            height: Height of the rectangle in pixels (minimum 1).
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).

        Returns:
            Confirmation message with the number of pixels drawn.
        """
        r, g, b = self._clamp_color(r, g, b)

        # Clamp rectangle to canvas bounds
        x1 = self._clamp(x, 0, self.width - 1)
        y1 = self._clamp(y, 0, self.height - 1)
        x2 = self._clamp(x + width, 0, self.width)
        y2 = self._clamp(y + height, 0, self.height)

        if x2 <= x1 or y2 <= y1:
            return "Rectangle out of bounds, nothing drawn."

        self.pixels[y1:y2, x1:x2] = [r, g, b]
        pixels_drawn = (x2 - x1) * (y2 - y1)
        return (
            f"Drew rectangle at ({x1},{y1}) size {x2 - x1}x{y2 - y1} "
            f"with RGB({r},{g},{b}). {pixels_drawn} pixels drawn."
        )

    def draw_line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        r: int,
        g: int,
        b: int,
        thickness: int = 2,
    ) -> str:
        """Draw a line between two points using Bresenham's algorithm with thickness.

        Great for outlines, edges, borders, and structural details.

        Args:
            x1: Start X coordinate (0-199).
            y1: Start Y coordinate (0-199).
            x2: End X coordinate (0-199).
            y2: End Y coordinate (0-199).
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).
            thickness: Line thickness in pixels (default 2, minimum 1).

        Returns:
            Confirmation message with the number of pixels drawn.
        """
        r, g, b = self._clamp_color(r, g, b)
        thickness = max(1, thickness)
        half_t = thickness // 2

        pixels_drawn = 0

        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        cx, cy = x1, y1

        while True:
            # Draw a square brush at the current point for thickness
            for bx in range(-half_t, half_t + 1):
                for by in range(-half_t, half_t + 1):
                    px = cx + bx
                    py = cy + by
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.pixels[py, px] = [r, g, b]
                        pixels_drawn += 1

            if cx == x2 and cy == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return (
            f"Drew line from ({x1},{y1}) to ({x2},{y2}) "
            f"thickness={thickness} RGB({r},{g},{b}). ~{pixels_drawn} pixels drawn."
        )

    def draw_circle(
        self,
        cx: int,
        cy: int,
        radius: int,
        r: int,
        g: int,
        b: int,
        fill: bool = True,
    ) -> str:
        """Draw a circle (filled or outlined) on the canvas.

        Useful for rounded features like heads, eyes, buttons, and decorative elements.

        Args:
            cx: Center X coordinate (0-199).
            cy: Center Y coordinate (0-199).
            radius: Radius in pixels (minimum 1).
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).
            fill: If True, draws a filled circle. If False, draws only the outline.

        Returns:
            Confirmation message with the number of pixels drawn.
        """
        r, g, b = self._clamp_color(r, g, b)
        radius = max(1, radius)
        pixels_drawn = 0

        if fill:
            # Filled circle using distance check
            for py in range(
                max(0, cy - radius), min(self.height, cy + radius + 1)
            ):
                for px in range(
                    max(0, cx - radius), min(self.width, cx + radius + 1)
                ):
                    if (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2:
                        self.pixels[py, px] = [r, g, b]
                        pixels_drawn += 1
        else:
            # Outlined circle using midpoint circle algorithm
            x = radius
            y_off = 0
            decision = 1 - radius

            while x >= y_off:
                # Draw 8 symmetry points
                for px, py in [
                    (cx + x, cy + y_off),
                    (cx - x, cy + y_off),
                    (cx + x, cy - y_off),
                    (cx - x, cy - y_off),
                    (cx + y_off, cy + x),
                    (cx - y_off, cy + x),
                    (cx + y_off, cy - x),
                    (cx - y_off, cy - x),
                ]:
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.pixels[py, px] = [r, g, b]
                        pixels_drawn += 1

                y_off += 1
                if decision <= 0:
                    decision += 2 * y_off + 1
                else:
                    x -= 1
                    decision += 2 * (y_off - x) + 1

        fill_str = "filled" if fill else "outlined"
        return (
            f"Drew {fill_str} circle at ({cx},{cy}) radius={radius} "
            f"RGB({r},{g},{b}). {pixels_drawn} pixels drawn."
        )

    def save(self, path: str) -> str:
        """Save the canvas as a PNG image file.

        Args:
            path: File path to save to (should end in .png).

        Returns:
            Confirmation message.
        """
        img = Image.fromarray(self.pixels, mode="RGB")
        img.save(path)
        return f"Canvas saved to {path}."

    def to_base64(self) -> str:
        """Encode the current canvas state as a base64 PNG data URI.

        Returns:
            A data URI string suitable for OpenAI vision API messages.
        """
        img = Image.fromarray(self.pixels, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def get_pixel_count(self) -> dict:
        """Get a summary of non-white pixels on the canvas (for progress tracking).

        Returns:
            Dictionary with total pixels and non-white pixel count.
        """
        total = self.width * self.height
        white = np.all(self.pixels == 255, axis=2).sum()
        drawn = total - int(white)
        return {"total_pixels": total, "drawn_pixels": drawn, "coverage_pct": round(drawn / total * 100, 1)}

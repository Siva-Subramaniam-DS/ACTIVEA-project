"""Visualization and artifact generation module for ACTIVEA.

Generates standalone, publication-quality SVG figures and Markdown summary tables
without requiring third-party libraries (pure Python standard library).
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
from pathlib import Path


class SvgChartGenerator:
    """Generates clean, professional SVG charts for reports and presentation slides."""

    @staticmethod
    def generate_convergence_plot(
        series_data: Dict[str, List[float]],
        title: str = "Algorithm Convergence Comparison",
        x_label: str = "Generation / Iteration Checkpoint",
        y_label: str = "Fitness Score",
        width: int = 800,
        height: int = 480,
    ) -> str:
        """Renders multi-line convergence trajectory comparison into an SVG string."""
        pad_left = 70
        pad_right = 240
        pad_top = 60
        pad_bottom = 60

        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        # Find min/max across all series
        all_vals = [v for s in series_data.values() for v in s]
        min_y = max(0.0, min(all_vals) * 0.95 if all_vals else 0.0)
        max_y = max(all_vals) * 1.05 if all_vals else 100.0
        max_x = max(len(s) for s in series_data.values()) - 1
        max_x = max(1, max_x)

        colors = [
            "#2563EB",  # Blue (Standard GA)
            "#DC2626",  # Red (Bespoke Repair GA)
            "#059669",  # Emerald (Differential Evolution)
            "#D97706",  # Amber (Particle Swarm Optimization)
            "#7C3AED",  # Violet (Simulated Annealing)
            "#0891B2",  # Cyan (Deterministic Crowding GA)
            "#DB2777",  # Pink ((1+1) Evolution Strategy)
            "#4F46E5",  # Indigo (Iterated Local Search)
            "#94A3B8",  # Slate (Random Search Baseline)
            "#0D9488",  # Teal (Extra)
        ]

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, '
            f'\'Segoe UI\', Roboto, sans-serif;">',
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
            f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="18" '
            f'font-weight="bold" fill="#1E293B">{title}</text>',
        ]

        # Axes & Gridlines
        svg_lines.append(
            f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
            f'y2="{pad_top + plot_h}" stroke="#94A3B8" stroke-width="2"/>'
        )
        svg_lines.append(
            f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
            f'y2="{pad_top + plot_h}" stroke="#94A3B8" stroke-width="2"/>'
        )

        num_y_ticks = 5
        for i in range(num_y_ticks + 1):
            y_val = min_y + (max_y - min_y) * (i / num_y_ticks)
            y_pos = pad_top + plot_h - (i / num_y_ticks) * plot_h
            svg_lines.append(
                f'<line x1="{pad_left}" y1="{y_pos}" x2="{pad_left + plot_w}" '
                f'y2="{y_pos}" stroke="#E2E8F0" stroke-dasharray="4,4"/>'
            )
            svg_lines.append(
                f'<text x="{pad_left - 10}" y="{y_pos + 4}" text-anchor="end" '
                f'font-size="11" fill="#64748B">{y_val:.1f}</text>'
            )

        # X Axis labels
        num_x_ticks = 6
        for i in range(num_x_ticks):
            x_idx = int(i * max_x / (num_x_ticks - 1))
            x_pos = pad_left + (x_idx / max_x) * plot_w
            svg_lines.append(
                f'<text x="{x_pos}" y="{pad_top + plot_h + 20}" text-anchor="middle" '
                f'font-size="11" fill="#64748B">{x_idx}</text>'
            )

        svg_lines.append(
            f'<text x="{pad_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#334155">{x_label}</text>'
        )
        svg_lines.append(
            f'<text x="20" y="{pad_top + plot_h / 2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {pad_top + plot_h / 2})" font-size="13" '
            f'font-weight="600" fill="#334155">{y_label}</text>'
        )

        # Plot Data Series & Legend
        legend_x = pad_left + plot_w + 15
        legend_y = pad_top + 20

        for s_idx, (name, values) in enumerate(series_data.items()):
            color = colors[s_idx % len(colors)]
            points = []
            for x_idx, v in enumerate(values):
                x_pos = pad_left + (x_idx / max_x) * plot_w
                y_pos = pad_top + plot_h - ((v - min_y) / max(1e-6, max_y - min_y)) * plot_h
                points.append(f"{x_pos:.1f},{y_pos:.1f}")

            points_str = " ".join(points)
            svg_lines.append(
                f'<polyline points="{points_str}" fill="none" stroke="{color}" '
                f'stroke-width="2.5" stroke-linecap="round"/>'
            )

            # Legend
            curr_leg_y = legend_y + s_idx * 26
            svg_lines.append(
                f'<line x1="{legend_x}" y1="{curr_leg_y}" x2="{legend_x + 20}" '
                f'y2="{curr_leg_y}" stroke="{color}" stroke-width="3"/>'
            )
            svg_lines.append(
                f'<text x="{legend_x + 26}" y="{curr_leg_y + 4}" font-size="12" '
                f'font-weight="500" fill="#1E293B">{name}</text>'
            )

        svg_lines.append("</svg>")
        return "\n".join(svg_lines)

    @staticmethod
    def save_svg(svg_content: str, filepath: str | Path) -> None:
        """Saves SVG string to the specified file path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg_content)

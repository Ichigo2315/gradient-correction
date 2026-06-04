"""Render a PPT-ready flowchart of the visual-servo simulator pipeline."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("servo_eval/visualizations/servo_simulator_flowchart.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# UCSD-ish palette
NAVY = "#182B49"
TEAL = "#00A0C6"
TEAL_FILL = "#e2f4fa"
BLUE_FILL = "#e8eef6"
CTRL_FILL = "#fdeee3"
ORANGE = "#e08a3c"
GRAY = "#5a5a5a"

fig, ax = plt.subplots(figsize=(7.2, 9.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")

CX = 4.3          # main column center
BW, BH = 5.4, 1.15  # box width / height


def box(y, title, subtitle, fill, edge, *, cx=CX, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=3))
    ax.text(cx, y + 0.17, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=NAVY, zorder=4)
    if subtitle:
        ax.text(cx, y - 0.27, subtitle, ha="center", va="center",
                fontsize=11, color=GRAY, zorder=4)


def varrow(y_from, y_to, *, cx=CX, color=NAVY, label=None):
    ax.add_patch(FancyArrowPatch(
        (cx, y_from), (cx, y_to),
        arrowstyle="-|>", mutation_scale=20,
        linewidth=2.0, color=color, zorder=2))
    if label:
        ax.text(cx + 0.18, (y_from + y_to) / 2, label, ha="left", va="center",
                fontsize=10.5, color=color, style="italic")


# ---- main top-down chain ----
y = [12.2, 10.4, 8.6, 6.8, 5.0, 3.2]
box(y[0], "Mask sequence  $\\hat{y}_t$",
    "AOT(ori)  /  AOT+GC   (indexed PNG)", TEAL_FILL, TEAL)
box(y[1], "Centroid extraction",
    "observation  $z_t$ = object centroid", BLUE_FILL, "#7fa8d0")
box(y[2], "Kalman filter  (const-velocity)",
    "smoothed state  $\\hat{x}_t = [c_x, c_y, v_x, v_y]$", BLUE_FILL, "#7fa8d0")
box(y[3], "Tracking error",
    "$e_t = \\hat{x}_t - c_t$   (camera-lead error)", CTRL_FILL, ORANGE)
box(y[4], "PID controller",
    "command  $u_t = \\mathrm{PID}(e_t)$", CTRL_FILL, ORANGE)
box(y[5], "Virtual pan-tilt camera",
    "$c_{t+1} = c_t + u_t\\,\\Delta t$", CTRL_FILL, ORANGE)

for a, b in zip(y[:-1], y[1:]):
    varrow(a - BH / 2, b + BH / 2)

# ---- closed-loop feedback: camera -> error ----
x_right = CX + BW / 2
fb_x = x_right + 1.05
ax.add_patch(FancyArrowPatch(
    (x_right, y[5]), (fb_x, y[5]), arrowstyle="-", linewidth=2.0,
    color=TEAL, zorder=2))
ax.add_patch(FancyArrowPatch(
    (fb_x, y[5]), (fb_x, y[3]), arrowstyle="-", linewidth=2.0,
    color=TEAL, zorder=2))
ax.add_patch(FancyArrowPatch(
    (fb_x, y[3]), (x_right, y[3]), arrowstyle="-|>", mutation_scale=20,
    linewidth=2.0, color=TEAL, zorder=2))
ax.text(fb_x + 0.12, (y[3] + y[5]) / 2, "feedback\n$c_t$", ha="left",
        va="center", fontsize=10.5, color=TEAL, fontweight="bold")

# ---- metrics output ----
my = 1.2
box(my, "Downstream physical metrics",
    "tracking RMSE · P99 · control energy · centroid jerk",
    NAVY, NAVY, w=7.2, h=1.05)
ax.texts[-1].set_color("#dfe7ef")          # subtitle lighter on navy
ax.texts[-2].set_color("white")            # title white on navy
varrow(y[5] - BH / 2, my + 0.55)
ax.text(CX + 0.2, (y[5] - BH / 2 + my + 0.55) / 2,
        "from $e_t,\\,u_t,\\,\\hat{x}_t$", ha="left", va="center",
        fontsize=10, color=GRAY, style="italic")

# ---- side bracket label: perception vs control ----
ax.text(CX - BW / 2 - 0.45, (y[1] + y[2]) / 2, "Perception", rotation=90,
        ha="center", va="center", fontsize=11, color="#5b7aa6",
        fontweight="bold")
ax.text(CX - BW / 2 - 0.45, (y[3] + y[5]) / 2, "Control loop", rotation=90,
        ha="center", va="center", fontsize=11, color=ORANGE,
        fontweight="bold")

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.with_suffix(".transparent.png"), dpi=220,
            bbox_inches="tight", transparent=True)
print(f"saved: {OUT}")

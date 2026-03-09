import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'wspace': 0.15})

# Colors
C_TEXT = '#607D8B'
C_AUDIO = '#1976D2'
C_IMAGE = '#388E3C'
C_BOTH = '#7B1FA2'
C_Q = '#E65100'

modalities = ['Text-only', 'Audio-only', 'Image-only', 'Both']
colors = [C_TEXT, C_AUDIO, C_IMAGE, C_BOTH]

example_q = 'Q: "How many instruments are playing in the video?"'

new_prefixes = [
    "Use only the question text and general world\nknowledge. Do not assume access to audio\nor visual evidence.",
    "Use only auditory evidence from the provided\naudio. Ignore visual priors and answer from what\nyou hear, such as timbre, pitch, rhythm, or\nsource sound.",
    "Use only visual evidence from the provided\nimage. Ignore audio priors and answer from what\nyou see, such as object appearance, motion,\ncount, or scene cues.",
    "Combine auditory and visual evidence. Use\nsound for audio properties, use vision for visual\nproperties, and resolve conflicts by requiring\nconsistency across both modalities.",
]

# ============== LEFT: Old Prompt ==============
ax = axes[0]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1, 11)
ax.axis('off')
ax.set_title('Old Prompt Strategy\n(No modality awareness)', fontsize=15,
             fontweight='bold', color='#424242', pad=20)

y_positions = [8.0, 5.5, 3.0, 0.5]
for i, (lab, col, yp) in enumerate(zip(modalities, colors, y_positions)):
    # Rounded box
    rect = mpatches.FancyBboxPatch((0.5, yp), 8.5, 1.5,
                                    boxstyle="round,pad=0.2",
                                    facecolor='white', edgecolor=col,
                                    linewidth=2.5)
    ax.add_patch(rect)
    # Modality label (top-left inside box)
    ax.text(1.0, yp + 1.2, lab, fontsize=12, fontweight='bold', color=col, va='center')
    # Question (centered)
    ax.text(4.75, yp + 0.5, example_q, fontsize=10, ha='center', va='center',
            color='#555', fontstyle='italic')

# Bracket showing all identical
ax.annotate('', xy=(9.5, 0.5), xytext=(9.5, 9.5),
            arrowprops=dict(arrowstyle='-', color='#D32F2F', lw=2.5,
                           connectionstyle='bar,fraction=-0.05'))
ax.text(10.5, 5.0, 'All\nidentical', fontsize=12, ha='left', va='center',
        color='#D32F2F', fontweight='bold')

# ============== RIGHT: New Prompt ==============
ax = axes[1]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1, 11)
ax.axis('off')
ax.set_title('New Prompt Strategy\n(Modality-aware prefixes)', fontsize=15,
             fontweight='bold', color='#6A1B9A', pad=20)

y_positions = [8.2, 5.6, 2.9, 0.2]
box_h = 2.3
for i, (lab, col, yp) in enumerate(zip(modalities, colors, y_positions)):
    # Rounded box
    rect = mpatches.FancyBboxPatch((0.5, yp), 9.0, box_h,
                                    boxstyle="round,pad=0.2",
                                    facecolor='#FAFAFA', edgecolor=col,
                                    linewidth=2.5)
    ax.add_patch(rect)

    # Modality label
    ax.text(1.0, yp + box_h - 0.25, lab, fontsize=12, fontweight='bold', color=col, va='center')

    # Prefix region (light tinted background)
    prefix_rect = mpatches.FancyBboxPatch((0.7, yp + 0.65), 8.6, box_h - 1.05,
                                           boxstyle="round,pad=0.1",
                                           facecolor=col, edgecolor='none',
                                           alpha=0.08)
    ax.add_patch(prefix_rect)

    # Prefix text
    ax.text(1.0, yp + box_h - 0.7, new_prefixes[i], fontsize=8.2,
            va='top', color=col, fontweight='bold', linespacing=1.4)

    # Separator
    ax.plot([0.9, 9.1], [yp + 0.55, yp + 0.55], '-', color='#BDBDBD', linewidth=0.8)

    # Question
    ax.text(5.0, yp + 0.25, example_q, fontsize=9.5, ha='center', va='center',
            color='#555', fontstyle='italic')

# Bracket showing all unique
ax.annotate('', xy=(9.9, 0.2), xytext=(9.9, 10.5),
            arrowprops=dict(arrowstyle='-', color='#6A1B9A', lw=2.5,
                           connectionstyle='bar,fraction=-0.03'))
ax.text(10.5, 5.3, 'Each\nunique', fontsize=12, ha='left', va='center',
        color='#6A1B9A', fontweight='bold')

plt.savefig('prompt_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('prompt_comparison.pdf', bbox_inches='tight')
print("Saved: prompt_comparison.png / .pdf")
plt.show()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# Music-AVQA composition results
modalities = ['Text', 'Audio-Text', 'Vision', 'All Modalities']
scores = [27.0, 41.0, 54.0, 59.0]
colors = ['#5B9BD5', '#A855F7', '#ED7D31', '#FFC000']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(modalities, scores, color=colors, width=0.55, edgecolor='white', linewidth=1.2)

# Add value labels on bars
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f'{score:.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Annotate the fusion gain
ax.annotate('+5.0 gain\nvs best single',
            xy=(3, 59), xytext=(2.15, 66),
            fontsize=10, ha='center', color='#333333',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF2CC', ec='#FFC000', lw=1.2))

ax.set_ylabel('Extracted Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Music-AVQA Composition — Modality Performance', fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, 80)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('audio_performance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('audio_performance.pdf', bbox_inches='tight', facecolor='white')
print("Saved: audio_performance.png and audio_performance.pdf")

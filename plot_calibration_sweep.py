import matplotlib.pyplot as plt
import numpy as np
# Manual log fit instead of scipy

# Data from calibration sweep logs (best epoch results)
# n = number of training samples
n_observed = [1, 5, 100]

# Final eval extracted_em scores per modality (best epoch)
audio_em_obs = [67.42, 67.42, 67.75]
image_em_obs = [73.51, 73.41, 73.23]
both_em_obs =  [60.41, 60.54, 64.18]

# --- Fit log curve to extrapolate: y = a * log(x+1) + b ---
# Using least squares via numpy: fit y = a * ln(x+1) + b
def fit_log(x_data, y_data):
    X = np.log(np.array(x_data) + 1)
    A = np.vstack([X, np.ones(len(X))]).T
    a, b = np.linalg.lstsq(A, y_data, rcond=None)[0]
    return a, b

def log_fit(x, a, b):
    return a * np.log(np.array(x) + 1) + b

popt_both = fit_log(n_observed, both_em_obs)
popt_audio = fit_log(n_observed, audio_em_obs)
popt_image = fit_log(n_observed, image_em_obs)

# Extrapolate to n=1000
n_extrap = np.array([1, 5, 100, 1000])
n_smooth = np.logspace(0, 3, 200)

both_extrap = log_fit(n_smooth, *popt_both)
audio_extrap = log_fit(n_smooth, *popt_audio)
image_extrap = log_fit(n_smooth, *popt_image)

# Point predictions at n=1000
both_1000 = log_fit(1000, *popt_both)
audio_1000 = log_fit(1000, *popt_audio)
image_1000 = log_fit(1000, *popt_image)
avg_ind_1000 = (audio_1000 + image_1000) / 2

print(f"Extrapolated n=1000: both={both_1000:.1f}, audio={audio_1000:.1f}, image={image_1000:.1f}")
print(f"Predicted composition gap at n=1000: {avg_ind_1000 - both_1000:.1f}%")

# All points including extrapolation
n_all = [1, 5, 100, 1000]
both_all = both_em_obs + [both_1000]
audio_all = audio_em_obs + [audio_1000]
image_all = image_em_obs + [image_1000]
avg_individual = [(a + i) / 2 for a, i in zip(audio_all, image_all)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left: Eval accuracy vs data volume with extrapolation ---
# Observed points (solid)
ax1.plot(n_observed, audio_em_obs, 'o', color='#2196F3', markersize=9, zorder=5)
ax1.plot(n_observed, image_em_obs, 's', color='#4CAF50', markersize=9, zorder=5)
ax1.plot(n_observed, both_em_obs, 'D', color='#F44336', markersize=10, zorder=5)

# Extrapolated curves
ax1.plot(n_smooth, audio_extrap, '-', color='#2196F3', linewidth=2, label='Audio-only')
ax1.plot(n_smooth, image_extrap, '-', color='#4CAF50', linewidth=2, label='Image-only')
ax1.plot(n_smooth, both_extrap, '-', color='#F44336', linewidth=2.5, label='Both (composition)')
ax1.plot(n_smooth, (audio_extrap + image_extrap) / 2, '--', color='gray', linewidth=1.5, alpha=0.7, label='Avg individual (ceiling)')

# Extrapolated point at n=1000 (hollow marker)
ax1.plot(1000, audio_1000, 'o', color='#2196F3', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=5)
ax1.plot(1000, image_1000, 's', color='#4CAF50', markersize=9, markerfacecolor='white', markeredgewidth=2, zorder=5)
ax1.plot(1000, both_1000, 'D', color='#F44336', markersize=10, markerfacecolor='white', markeredgewidth=2, zorder=5)

# Shade extrapolation region
ax1.axvspan(100, 1000, alpha=0.08, color='orange', label='Extrapolated region')

ax1.set_xscale('log')
ax1.set_xlabel('Training Samples (log scale)', fontsize=12)
ax1.set_ylabel('Extracted EM (%)', fontsize=12)
ax1.set_title('Calibration Sweep: Accuracy vs Data Volume\n(with log-fit extrapolation to n=1000)', fontsize=13, fontweight='bold')
ax1.set_xticks([1, 5, 100, 1000])
ax1.set_xticklabels(['1', '5', '100', '1000\n(extrap.)'])
ax1.set_ylim(55, 78)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)

# Annotate composition gap at each point
for i, n in enumerate(n_all):
    gap = avg_individual[i] - both_all[i]
    style = 'italic' if n == 1000 else 'normal'
    ax1.annotate(f'gap={gap:.1f}',
                 xy=(n, both_all[i]), xytext=(0, -18),
                 textcoords='offset points', ha='center', fontsize=9,
                 color='#F44336', fontweight='bold', fontstyle=style)

# --- Right: Composition gap & gain vs data volume ---
gaps = [avg - b for avg, b in zip(avg_individual, both_all)]
gain = [both_all[i] - both_all[0] for i in range(len(n_all))]

ax2_color = '#F44336'
colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350']
hatches = ['', '', '', '//']
bars = ax2.bar(range(len(n_all)), gaps, color=colors,
               edgecolor=ax2_color, linewidth=1.5, width=0.5)
bars[3].set_hatch('//')  # mark extrapolated

ax2.set_xticks(range(len(n_all)))
ax2.set_xticklabels([f'n={n}\n({n}x)' for n in n_all])
ax2.set_ylabel('Composition Gap (avg individual - both) %', fontsize=11, color=ax2_color)
ax2.set_title('Composition Gap Persists Despite\nPolynomial Data Increase', fontsize=13, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=ax2_color)
ax2.grid(True, alpha=0.3, axis='y')

# Gain on secondary axis
ax2b = ax2.twinx()
ax2b.plot(range(len(n_all)), gain, 'ko--', markersize=8, linewidth=2, label='Composition gain')
ax2b.plot(3, gain[3], 'o', color='black', markersize=8, markerfacecolor='white', markeredgewidth=2)
ax2b.set_ylabel('Composition EM Gain over n=1 (%)', fontsize=11)
ax2b.set_ylim(-1, 12)

for i, g in enumerate(gain):
    suffix = ' (extrap.)' if i == 3 else ''
    ax2b.annotate(f'+{g:.1f}%{suffix}', xy=(i, g), xytext=(0, 10),
                  textcoords='offset points', ha='center', fontsize=9.5, fontweight='bold')

# Summary box
textstr = (f'1000x more data -> ~+{gain[3]:.1f}% gain (extrap.)\n'
           f'Composition gap at n=1000: ~{gaps[3]:.1f}%\n'
           'Log-scaling: diminishing returns\n'
           '-> Not a calibration problem')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9)
ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', ha='center', bbox=props)

plt.tight_layout()
plt.savefig('calibration_sweep_results.png', dpi=150, bbox_inches='tight')
plt.savefig('calibration_sweep_results.pdf', bbox_inches='tight')
print("Saved: calibration_sweep_results.png / .pdf")
plt.show()

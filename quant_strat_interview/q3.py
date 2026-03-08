import numpy as np
import matplotlib.pyplot as plt
# a
def volatility_binary(correlation, w1=0.5, w2=0.5, sigma1=0.2, sigma2=0.3):
    return np.sqrt((w1*sigma1)**2 + (w2*sigma2)**2 + 2*w1*w2*sigma1*sigma2*correlation)


# 2. Define the correlation range (rho) from -1 (-100%) to +1 (+100%)
correlation = np.linspace(-1, 1, 500)
vol_p = volatility_binary(correlation)

# 4. Create the professional plot
plt.figure(figsize=(10, 6))
plt.plot(correlation, vol_p, label='Portfolio Volatility ($\sigma_p$)', color='#1f77b4', linewidth=2.5)

# 5. Highlight boundary and midpoint values
rho_points = np.array([-1, 0, 1])
vol_points = volatility_binary(rho_points)

# Mark the points
plt.scatter(rho_points, vol_points, color='red', zorder=5)

# Label the points (in percentage format)
for r, v in zip(rho_points, vol_points):
    plt.annotate(f'({r*100:+.0f}%, {v*100:.1f}%)', 
                 xy=(r, v), xytext=(5, 5) if r < 1 else (-45, -15),
                 textcoords='offset points', fontsize=10, fontweight='bold')

# 6. Chart formatting for a professional report
plt.title('2-Asset Portfolio Volatility as a Function of Correlation', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient ($\\rho$)', fontsize=12)
plt.ylabel('Portfolio Volatility ($\sigma_p$)', fontsize=12)

# Convert axes ticks to percentages
plt.xticks(np.arange(-1, 1.1, 0.2), [f'{x*100:.0f}%' for x in np.arange(-1, 1.1, 0.2)])
plt.yticks(np.arange(0, 0.31, 0.05), [f'{y*100:.0f}%' for y in np.arange(0, 0.31, 0.05)])

plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(0, color='black', linewidth=1)
plt.ylim(0, 0.35)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

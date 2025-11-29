import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Data Entry (From Screenshot) ---
# Favorable Gradient (lambda >= 0)
lam_fav = np.array([0.25, 0.20, 0.14, 0.12, 0.10, 0.080, 0.064, 0.048, 0.032, 0.016, 0.0])
H_fav   = np.array([2.00, 2.07, 2.18, 2.23, 2.28, 2.34, 2.39, 2.44, 2.49, 2.55, 2.61])

# Adverse Gradient (lambda < 0)
lam_adv = np.array([
    -0.016, -0.032, -0.040, -0.048, -0.052, -0.056, -0.060, 
    -0.064, -0.068, -0.072, -0.076, -0.080, -0.084, -0.086, -0.088, -0.090
])
H_adv   = np.array([
    2.67, 2.75, 2.81, 2.87, 2.90, 2.94, 2.99, 
    3.04, 3.09, 3.15, 3.22, 3.30, 3.39, 3.44, 3.49, 3.55
])

def poly2(x, a, b, c):
    return a + b*x + c*x**2

def rational_func(x, a, b, c):
    return a + b / (c + x)

popt_H_fav, _ = curve_fit(poly2, lam_fav, H_fav)
print("lambda >= 0:")
print(f"H(lambda) = {popt_H_fav[0]:.3f} + {popt_H_fav[1]:.3f}*lambda + {popt_H_fav[2]:.3f}*lambda^2")

initial_guess = [2.0, 0.07, 0.14] 
popt_H_adv, _ = curve_fit(rational_func, lam_adv, H_adv, p0=initial_guess)

print("lambda < 0:")
print(f"H(lambda) = {popt_H_adv[0]:.4f} + {popt_H_adv[1]:.4f} / ({popt_H_adv[2]:.4f} + lambda)")

l_fav_smooth = np.linspace(0, 0.25, 100)
H_fav_fit = poly2(l_fav_smooth, *popt_H_fav)

l_adv_smooth = np.linspace(-0.09, 0, 100)
H_adv_fit = rational_func(l_adv_smooth, *popt_H_adv)

plt.figure(figsize=(10, 6))


plt.scatter(lam_fav, H_fav, color='blue', label='Data')
plt.scatter(lam_adv, H_adv, color='red', label='Data')


plt.plot(l_fav_smooth, H_fav_fit, 'b--', linewidth=2, label='Fit')
plt.plot(l_adv_smooth, H_adv_fit, 'r--', linewidth=2, label='Fit')

plt.title('Curve Fitting for Shape Factor ')
plt.xlabel('$\lambda$')
plt.ylabel('Shape Factor $H$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
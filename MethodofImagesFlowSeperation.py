import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

U_real = 10.0
rho = 1.225
nu = 1.51e-5

r_b_m = 10.0 / 1000.0
r_t_m = 10.0 / 1000.0
h_0_m = 5.0 / 1000.0
D_m = r_b_m + r_t_m + h_0_m 

theta = np.linspace(np.pi - 0.01, 0.01, 400) 


z_surface = r_b_m * np.exp(1j * theta)

s = r_b_m * (np.pi - theta) 

a = np.pi / (2 * D_m)
sinh_term = np.sinh(a * z_surface)
cosh_term = np.cosh(a * z_surface)
csch_sq = 1 / (sinh_term**2)
sech_sq = 1 / (cosh_term**2)

w_z_norm = (1 - (a * r_b_m)**2 * csch_sq + (a * r_t_m)**2 * sech_sq)
U_e = np.abs(w_z_norm) * U_real

U_e[0] = 0.0

integrand = U_e**5

integral_Ue5 = cumulative_trapezoid(integrand, s, initial=0)

epsilon = 1e-9
theta_sq = (0.45 * nu / (U_e**6 + epsilon)) * integral_Ue5

dUe_ds = np.gradient(U_e, s)

lambda_param = (theta_sq / nu) * dUe_ds

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.plot(s * 1000, U_e, 'b-', label='$U_e$')
ax1.set_ylabel('Velocity (m/s)', color='b')
ax1.set_title('Boundary Layer Analysis')
ax1.grid(True)

peak_idx = np.argmax(U_e)
ax1.axvline(s[peak_idx]*1000, color='g', linestyle='--', label='Peak Velocity')

ax2.plot(s * 1000, lambda_param, 'purple', label='Thwaites $\lambda$')
ax2.axhline(-0.09, color='r', linestyle='--', label='Separation Criterion (-0.09)')
ax2.set_ylabel('$\lambda$')
ax2.set_xlabel('Arc Length $s$ from Stagnation (mm)')
ax2.set_ylim(-0.2, 0.2)
ax2.grid(True)

sep_idx = np.where(lambda_param <= -0.09)[0]
if len(sep_idx) > 0:
    valid_sep = [idx for idx in sep_idx if s[idx] > 0.001]
    if valid_sep:
        first_sep = valid_sep[0]
        sep_s = s[first_sep] * 1000
        ax2.axvline(sep_s, color='k', linestyle='-.', label=f'Separation @ {sep_s:.1f} mm')

ax2.legend()
plt.tight_layout()
plt.show()
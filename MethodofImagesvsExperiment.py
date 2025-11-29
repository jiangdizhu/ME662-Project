import numpy as np
import matplotlib.pyplot as plt


r_b_cm = 0.15
r_t_cm = 0.15
h_0_cm = 0.04
D_cm = r_b_cm + r_t_cm + h_0_cm

r_b = r_b_cm * 10
r_t = r_t_cm * 10
h_0 = h_0_cm * 10
D = D_cm * 10
U = 1.0

x = np.linspace(-3 * D, 3 * D, 400)
y = np.linspace(0, D, 200)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

a = np.pi / (2 * D)
epsilon = 1e-9
Z_safe = X + 1j * (Y + epsilon)
sinh_term = np.sinh(a * Z_safe)
cosh_term = np.cosh(a * Z_safe)
csch_sq = 1 / (sinh_term**2)
sech_sq = 1 / (cosh_term**2)

w_z = U * (1 - (a * r_b)**2 * csch_sq + (a * r_t)**2 * sech_sq)
speed = np.abs(w_z)
Cp = 1 - (speed / U)**2
u_component = np.real(w_z)
v_component = -np.imag(w_z)

C_b = (np.pi * U * r_b**2) / (2 * D)
C_t = (np.pi * U * r_t**2) / (2 * D)
W_z = U * Z_safe + C_b * (cosh_term / sinh_term) + C_t * (sinh_term / cosh_term)
Psi = np.imag(W_z)

r_bottom = np.sqrt(X**2 + Y**2)
mask_b = (r_bottom <= r_b)
r_top = np.sqrt(X**2 + (Y - D)**2)
mask_t = (r_top <= r_t)
mask_all = mask_b | mask_t

speed[mask_all] = np.nan
Cp[mask_all] = np.nan
Psi[mask_all] = np.nan
u_component[mask_all] = np.nan
v_component[mask_all] = np.nan


exp_axial_dist_cm = np.array([
    0.0, 0.1, 0.21, 0.25, 0.29, 0.33, 0.37, 0.41, 0.46, 0.51, 0.54
])
exp_entrance_x = 0.21

exp_data = {
    "3 cm H2O": {
        "data": np.array([0.0, 0.1, 4.2, 3.1, 3.0, 2.9, 2.9, 2.9, 3.0, 2.8, 2.8]),
        "max_drop": 4.2,
        "color": "blue"
    },
    "5 cm H2O": {
        "data": np.array([0.0, 0.2, 6.5, 4.9, 4.8, 4.7, 4.6, 4.6, 4.7, 4.5, 4.5]),
        "max_drop": 6.5,
        "color": "green"
    },
    "10 cm H2O": {
        "data": np.array([0.0, 0.5, 12.8, 9.8, 9.0, 8.5, 8.4, 8.3, 8.4, 8.2, 8.3]),
        "max_drop": 12.8,
        "color": "red"
    },
    "15 cm H2O": {
        "data": np.array([0.0, 0.3, 18.5, 15.3, 15.2, 14.9, 14.8, 14.8, 15.0, 14.5, 14.5]),
        "max_drop": 18.5,
        "color": "purple"
    }
}

theta = np.linspace(0.01, np.pi - 0.01, 200)
x_model = r_b * np.cos(np.pi - theta)
z_surface = r_b * np.exp(1j * theta)

sinh_term_surf = np.sinh(a * z_surface)
cosh_term_surf = np.cosh(a * z_surface)
csch_sq_surf = 1 / (sinh_term_surf**2)
sech_sq_surf = 1 / (cosh_term_surf**2)
w_z_surf = U * (1 - (a * r_b)**2 * csch_sq_surf + (a * r_t)**2 * sech_sq_surf)
speed_surf = np.abs(w_z_surf)
Cp_surf = 1 - (speed_surf / U)**2
model_min_Cp = np.min(Cp_surf)


x_model_cm = x_model / 10.0
x_model_aligned = x_model_cm + exp_entrance_x

fig, ax = plt.subplots(figsize=(12, 8))

for case_name, info in exp_data.items():
    
    exp_drops = info["data"]
    exp_max_drop = info["max_drop"]
    color = info["color"]
    
    ax.plot(exp_axial_dist_cm, exp_drops, 's', color=color, markersize=8, label=f'Data ({case_name})', alpha=0.7)
    
    k_scale = exp_max_drop / (-model_min_Cp)
    model_pressure_drop = -Cp_surf * k_scale
    
    ax.plot(x_model_aligned, model_pressure_drop, '--', color=color, linewidth=2, label=f'Model ({case_name})')

ax.axvline(exp_entrance_x, color='gray', linestyle=':', label=f'Glottal Entrance (x={exp_entrance_x} cm)')
ax.axvline(exp_entrance_x + 0.3, color='gray', linestyle=':', label='Glottal Exit') # Lg=0.3cm

ax.set_title('Comparison of Simplified Model vs. All Experimental Cases', fontsize=16)
ax.set_xlabel('Axial Distance (cm)', fontsize=12)
ax.set_ylabel('Pressure Drop from Trachea (cm H₂O)', fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::2] + handles[1::2], labels[::2] + labels[1::2], fontsize=11, ncol=2)
ax.grid(True)

ax.invert_yaxis()
ax.set_ylim(20, -1)
ax.set_xlim(0, 0.7)

plt.tight_layout()

plt.show()
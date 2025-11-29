import numpy as np
import matplotlib.pyplot as plt

U = 1.0
r_b = 10.0
r_t = 10.0
h_0 = 5.0
D = r_b + r_t + h_0

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


bump_b_x = np.linspace(-r_b, r_b, 100)
bump_b_y = np.sqrt(r_b**2 - bump_b_x**2)
bump_t_x = np.linspace(-r_t, r_t, 100)
bump_t_y = D - np.sqrt(r_t**2 - bump_t_x**2)

fig, ax = plt.subplots(figsize=(12, 6))
levels = np.linspace(0, 3.0, 100)
contour = ax.contourf(X, Y, speed, levels=levels, cmap='jet', extend='max')
fig.colorbar(contour, label='Speed / U')

ax.streamplot(X, Y, u_component, v_component, color='white', density=1.5, linewidth=0.7, arrowstyle='->')

ax.fill(bump_b_x, bump_b_y, color='lightgrey', edgecolor='black')
ax.fill(bump_t_x, bump_t_y, color='lightgrey', edgecolor='black')

ax.set_title(f'Velocity Field & Streamlines (D={D}, $r_b$=$r_t$={r_t})', fontsize=14)
ax.set_xlabel('$x$ (mm)', fontsize=12)
ax.set_ylabel('$y$ (mm)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-2*D, 2*D)
ax.set_ylim(0, D)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(12, 6))
levels = np.linspace(-8.0, 1.0, 100)
contour = ax.contourf(X, Y, Cp, levels=levels, cmap='coolwarm', extend='min')
fig.colorbar(contour, label='$C_p$')

ax.fill(bump_b_x, bump_b_y, color='lightgrey', edgecolor='black')
ax.fill(bump_t_x, bump_t_y, color='lightgrey', edgecolor='black')

ax.set_title(f'Pressure Coefficient ($C_p$) Field (D={D}, $r_b$=$r_t$={r_t})', fontsize=14)
ax.set_xlabel('$x$ (mm)', fontsize=12)
ax.set_ylabel('$y$ (mm)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-2*D, 2*D)
ax.set_ylim(0, D)
plt.tight_layout()

midpoint_index = len(y) // 2

speed_centerline = speed[midpoint_index, :]
Cp_centerline = Cp[midpoint_index, :]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, speed_centerline, color='blue')
ax.set_title('Velocity Profile on Channel Centerline', fontsize=14)
ax.set_xlabel('$x$ (mm)', fontsize=12)
ax.set_ylabel('Speed / U', fontsize=12)
ax.grid(True)
ax.set_xlim(-3*D, 3*D)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, Cp_centerline, color='red')
ax.set_title('Pressure Coefficient $C_p$ on Channel Centerline', fontsize=14)
ax.set_xlabel('$x$ (mm)', fontsize=12)
ax.set_ylabel('$C_p$', fontsize=12)
ax.grid(True)
ax.set_xlim(-3*D, 3*D)
plt.tight_layout()

plt.show()
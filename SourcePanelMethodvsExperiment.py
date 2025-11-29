import numpy as np
import matplotlib.pyplot as plt

class Panel:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.xc = (x0 + x1) / 2.0
        self.yc = (y0 + y1) / 2.0
        self.L = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        self.phi = np.arctan2(y1 - y0, x1 - x0)
        self.nx = -np.sin(self.phi)
        self.ny = np.cos(self.phi)
        self.tx = np.cos(self.phi)
        self.ty = np.sin(self.phi)
        self.sigma = 0.0
        self.Vt = 0.0
        self.Cp = 0.0

class SourcePanelMethod:
    def __init__(self, panels, U_inf=100.0):
        self.panels = panels
        self.n_panels = len(panels)
        self.U_inf = U_inf
        
    def get_influence(self, target_idx, source_idx):
        p_i = self.panels[target_idx]
        p_j = self.panels[source_idx]
        if target_idx == source_idx:
            return 0.5, 0.0
        dx = p_i.xc - p_j.x0
        dy = p_i.yc - p_j.y0
        xi = dx * np.cos(p_j.phi) + dy * np.sin(p_j.phi)
        eta = -dx * np.sin(p_j.phi) + dy * np.cos(p_j.phi)
        r1_sq = np.maximum(xi**2 + eta**2, 1e-12)
        r2_sq = np.maximum((xi - p_j.L)**2 + eta**2, 1e-12)
        ln_term = 0.5 * np.log(r2_sq / r1_sq)
        theta1 = np.arctan2(eta, xi)
        theta2 = np.arctan2(eta, xi - p_j.L)
        d_theta = theta2 - theta1
        u_xi = (1.0 / (2 * np.pi)) * ln_term
        u_eta = (1.0 / (2 * np.pi)) * d_theta
        ug = u_xi * np.cos(p_j.phi) - u_eta * np.sin(p_j.phi)
        vg = u_xi * np.sin(p_j.phi) + u_eta * np.cos(p_j.phi)
        return (ug * p_i.nx + vg * p_i.ny), (ug * p_i.tx + vg * p_i.ty)

    def solve(self):
        N = self.n_panels
        A = np.zeros((N, N))
        b = np.zeros(N)
        for i in range(N):
            b[i] = - self.U_inf * self.panels[i].nx
            for j in range(N):
                vn, _ = self.get_influence(i, j)
                A[i, j] = vn
        sigmas = np.linalg.solve(A, b)
        for i in range(N):
            self.panels[i].sigma = sigmas[i]
            Vt_total = self.U_inf * self.panels[i].tx
            for j in range(N):
                _, vt = self.get_influence(i, j)
                Vt_total += vt * sigmas[j]
            self.panels[i].Vt = Vt_total
            self.panels[i].Cp = 1.0 - (Vt_total / self.U_inf)**2

def build_scherer_geometry_tuned():
    panels = []
    
    R = 0.15
    g = 0.04
    L_inlet = 0.5
    L_outlet = 1.0
    theta = np.radians(10)
    
    x_points = []
    y_points = []
    
    dx_in = L_inlet / 20
    for x in np.arange(-L_inlet, -R, dx_in):
        x_points.append(x)
        y_points.append(-(g/2 + R))
        
    n_curve = 50
    for alpha in np.linspace(np.pi, np.pi/2, n_curve):
        xc = R * np.cos(alpha)
        yc = -(g/2 + R) + R * np.sin(alpha)
        x_points.append(xc)
        y_points.append(yc)
        
    n_div = 60
    for x in np.linspace(0, L_outlet, n_div)[1:]:
        x_points.append(x)
        y_points.append(-(g/2 + x * np.tan(theta)))
        
    for k in range(len(x_points)-1):
        panels.append(Panel(x_points[k], y_points[k], x_points[k+1], y_points[k+1]))
        
    x_top = x_points[::-1]
    y_top = [-y for y in y_points[::-1]]
    for k in range(len(x_top)-1):
        panels.append(Panel(x_top[k], y_top[k], x_top[k+1], y_top[k+1]))
        
    return panels

panels = build_scherer_geometry_tuned()
solver = SourcePanelMethod(panels, U_inf=100.0)
solver.solve()

wall_panels = [p for p in panels if p.yc < 0]
wall_panels.sort(key=lambda p: p.xc)
x_model = np.array([p.xc for p in wall_panels])
Cp_model = np.array([p.Cp for p in wall_panels])
min_Cp_model = np.min(Cp_model)

exp_entrance_x = 0.21
exp_axial_dist_cm = np.array([0.0, 0.1, 0.21, 0.25, 0.29, 0.33, 0.37, 0.41, 0.46, 0.51, 0.54])

exp_data = {
    "3 cm H2O":  {"data": np.array([0.0, 0.1, 4.2, 3.1, 3.0, 2.9, 2.9, 2.9, 3.0, 2.8, 2.8]), "max_drop": 4.2, "color": "blue"},
    "5 cm H2O":  {"data": np.array([0.0, 0.2, 6.5, 4.9, 4.8, 4.7, 4.6, 4.6, 4.7, 4.5, 4.5]), "max_drop": 6.5, "color": "green"},
    "10 cm H2O": {"data": np.array([0.0, 0.5, 12.8, 9.8, 9.0, 8.5, 8.4, 8.3, 8.4, 8.2, 8.3]), "max_drop": 12.8, "color": "red"},
    "15 cm H2O": {"data": np.array([0.0, 0.3, 18.5, 15.3, 15.2, 14.9, 14.8, 14.8, 15.0, 14.5, 14.5]), "max_drop": 18.5, "color": "purple"}
}

fig, ax = plt.subplots(figsize=(12, 8))

x_model_aligned = x_model + exp_entrance_x

for case_name, info in exp_data.items():
    exp_drops = info["data"]
    exp_max_drop = info["max_drop"]
    color = info["color"]
    
    ax.plot(exp_axial_dist_cm, exp_drops, 's', color=color, markersize=8, label=f'Data ({case_name})', alpha=0.7)
    
    k_scale = exp_max_drop / (-min_Cp_model)
    model_pressure_drop = -Cp_model * k_scale
    
    ax.plot(x_model_aligned, model_pressure_drop, '--', color=color, linewidth=2, label=f'Panel Method ({case_name})')

ax.axvline(exp_entrance_x, color='gray', linestyle=':', label='Glottal Entrance')
ax.axvline(exp_entrance_x + 0.3, color='k', linestyle=':', label='Glottal Exit')

ax.set_title('Source Panel Method vs. Experimental Data (10-deg Divergence)', fontsize=16)
ax.set_xlabel('Axial Distance (cm)', fontsize=12)
ax.set_ylabel('Pressure Drop from Trachea (cm H₂O)', fontsize=12)
ax.invert_yaxis()
ax.grid(True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::2] + handles[1::2], labels[::2] + labels[1::2], fontsize=10, ncol=2)

plt.tight_layout()
plt.show()

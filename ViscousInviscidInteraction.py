import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
                
        try:
            sigmas = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sigmas = np.linalg.lstsq(A, b, rcond=None)[0]
            
        for i in range(N):
            self.panels[i].sigma = sigmas[i]
            Vt_total = self.U_inf * self.panels[i].tx
            for j in range(N):
                _, vt = self.get_influence(i, j)
                Vt_total += vt * sigmas[j]
            self.panels[i].Vt = Vt_total
            self.panels[i].Cp = 1.0 - (Vt_total / self.U_inf)**2

def calculate_shape_factor_H(lam):
    H = np.zeros_like(lam)
    
    mask_fav = lam >= 0
    l_fav = lam[mask_fav]
    H[mask_fav] = 2.61 - 3.836*l_fav + 5.607*l_fav**2
    
    mask_adv = lam < 0
    l_adv = lam[mask_adv]
    l_adv = np.maximum(l_adv, -0.139) 
    H[mask_adv] = 2.106 + 0.068 / (0.137 + l_adv)
    
    return H

def solve_thwaites(s, Ue, nu=0.15):
    n = len(s)
    
    Ue_safe = np.maximum(np.abs(Ue), np.max(np.abs(Ue)) * 0.01)
    
    integrand = Ue_safe**5
    integral = np.zeros(n)
    for i in range(1, n):
        ds = s[i] - s[i-1]
        integral[i] = integral[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * ds
        
    theta2 = (0.45 * nu * integral) / (Ue_safe**6)
    theta = np.sqrt(theta2)
    
    dUds = np.gradient(Ue_safe, s)
    lam = (theta2 / nu) * dUds
    
    H = calculate_shape_factor_H(lam)
    
    delta_star = H * theta
    
    sep_indices = np.where(lam <= -0.09)[0]
    
    valid_sep_idx = None
    if len(sep_indices) > 0:
        for idx in sep_indices:
            if s[idx] > s[-1]*0.1: 
                valid_sep_idx = idx
                break
                
    if valid_sep_idx is not None:
        idx = valid_sep_idx
        ds = s[idx] - s[idx-1]
        if ds > 1e-6:
            growth_rate = (delta_star[idx] - delta_star[idx-1]) / ds
            growth_rate = np.clip(growth_rate, -0.5, 0.5)
        else:
            growth_rate = 0
            
        for i in range(idx + 1, n):
            delta_star[i] = delta_star[idx] + growth_rate * (s[i] - s[idx])
            
    return delta_star, valid_sep_idx

def build_smooth_geometry_coordinates():
    R = 0.15          
    g = 0.04          
    L_inlet = 0.5     
    L_outlet = 1.0    
    theta = np.radians(10) 
    
    x_points = []
    y_points = []
    
    H_in = g/2 + R    
    H_th = g/2        
    
    dx_in = L_inlet / 25
    for x in np.arange(-L_inlet - R, -R, dx_in):
        x_points.append(x)
        y_points.append(-H_in)
        
    n_curve = 40
    for x in np.linspace(-R, 0, n_curve):
        xi = (x - (-R)) / (0 - (-R))
        
        h_local = H_in + (H_th - H_in) * (3*xi**2 - 2*xi**3)
        
        x_points.append(x)
        y_points.append(-h_local)
        
    n_div = 50
    for x in np.linspace(0, L_outlet, n_div)[1:]:
        x_points.append(x)
        y_points.append(-(H_th + x * np.tan(theta)))
        
    return np.array(x_points), np.array(y_points)

def make_panels_from_points(x_pts, y_pts):
    panels = []
    for k in range(len(x_pts)-1):
        panels.append(Panel(x_pts[k], y_pts[k], x_pts[k+1], y_pts[k+1]))
        
    x_top = x_pts[::-1]
    y_top = -y_pts[::-1]
    for k in range(len(x_top)-1):
        panels.append(Panel(x_top[k], y_top[k], x_top[k+1], y_top[k+1]))
        
    return panels

def run_vii_simulation():
    nu_air = 0.15 
    U_inf = 100.0 
    max_iter = 15
    relaxation = 0.05 
    tolerance = 1e-4
    
    x_orig, y_orig = build_smooth_geometry_coordinates()
    x_eff, y_eff = x_orig.copy(), y_orig.copy()
    
    cp_history = []
    delta_star_final = None
        
    for iteration in range(max_iter):
        panels = make_panels_from_points(x_eff, y_eff)
        
        solver = SourcePanelMethod(panels, U_inf)
        solver.solve()
        
        bot_panels = [p for p in panels if p.yc < 0]
        bot_panels.sort(key=lambda p: p.xc)
        
        xc_p = np.array([p.xc for p in bot_panels])
        ue_p = np.array([abs(p.Vt) for p in bot_panels]) 
        s_p = np.zeros_like(ue_p)
        
        curr_s = 0
        for i in range(len(bot_panels)):
            s_p[i] = curr_s + bot_panels[i].L / 2.0
            curr_s += bot_panels[i].L
            
        delta_star_centers, _ = solve_thwaites(s_p, ue_p, nu_air)
        
        s_nodes = np.zeros(len(x_eff))
        dist = np.sqrt(np.diff(x_eff)**2 + np.diff(y_eff)**2)
        s_nodes[1:] = np.cumsum(dist)
        
        f_delta = interp1d(
            s_p, 
            delta_star_centers, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(delta_star_centers[0], delta_star_centers[-1])
        )
        d_star_nodes = f_delta(s_nodes)
        
        d_star_smooth = d_star_nodes.copy()
        for _ in range(5): 
            d_star_smooth = np.convolve(d_star_smooth, np.ones(5)/5, mode='same')
            d_star_smooth[0:3] = d_star_nodes[0:3] 
            d_star_smooth[-3:] = d_star_nodes[-3:]

        max_allowed_disp = np.abs(y_orig) - 0.005 
        d_star_smooth = np.minimum(d_star_smooth, max_allowed_disp)
        
        y_target = y_orig + d_star_smooth
        x_target = x_orig 
        
        y_eff_next = (1 - relaxation) * y_eff + relaxation * y_target
        
        change = np.max(np.abs(y_eff_next - y_eff))
        
        x_eff = x_orig 
        y_eff = y_eff_next
        delta_star_final = d_star_smooth
        
        cp_iter = [p.Cp for p in bot_panels]
        cp_history.append(cp_iter)
        
        if change < tolerance:
            break

    return x_orig, y_orig, x_eff, y_eff, delta_star_final, cp_history

x_orig, y_orig, x_eff, y_eff, d_star, cp_hist = run_vii_simulation()

plt.figure(figsize=(12, 14))

plt.subplot(3, 1, 1)
plt.title("Viscous-Inviscid Interaction: Effective Geometry")
plt.plot(x_orig, y_orig, 'k-', linewidth=2, label="Physical Wall")
plt.plot(x_orig, -y_orig, 'k-', linewidth=2) 

plt.plot(x_eff, y_eff, 'r--', linewidth=2, label="Effective Wall (Displaced)")
plt.plot(x_eff, -y_eff, 'r--', linewidth=2)

plt.fill_between(x_orig, y_orig, y_eff, color='blue', alpha=0.3, label="Boundary Layer $\delta^*$")
plt.fill_between(x_orig, -y_orig, -y_eff, color='blue', alpha=0.3)

plt.axis('equal')
plt.xlabel("Axial Distance (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Evolution of Pressure Coefficient ($C_p$)")

xc_plot = (x_eff[:-1] + x_eff[1:]) / 2.0

plt.plot(xc_plot, cp_hist[0], 'k:', linewidth=1.5, label="Inviscid (Iter 0)")
if len(cp_hist) > 5:
    plt.plot(xc_plot, cp_hist[4], 'g--', linewidth=1, alpha=0.7, label="Iter 5")
plt.plot(xc_plot, cp_hist[-1], 'b-', linewidth=2, label="Final VII (Converged)")

plt.gca().invert_yaxis()
plt.xlabel("Axial Distance (cm)")
plt.ylabel("$C_p$")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Boundary Layer Displacement Thickness $\delta^*$")
plt.plot(x_orig, d_star, 'm-', linewidth=2, label="$\delta^*$")
plt.xlabel("Axial Distance (cm)")
plt.ylabel("$\delta^*$ (cm)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
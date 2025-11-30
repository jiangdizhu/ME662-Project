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
    
    def __init__(self, panels, U_inf=1000.0):
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
        

        r1_sq = xi**2 + eta**2
        r2_sq = (xi - p_j.L)**2 + eta**2
        
        r1_sq = max(r1_sq, 1e-12)
        r2_sq = max(r2_sq, 1e-12)
        
        ln_term = 0.5 * np.log(r2_sq / r1_sq)
        
        theta1 = np.arctan2(eta, xi)
        theta2 = np.arctan2(eta, xi - p_j.L)
        d_theta = theta2 - theta1
        
        u_xi_local = (1.0 / (2 * np.pi)) * ln_term
        u_eta_local = (1.0 / (2 * np.pi)) * d_theta
       
        ug = u_xi_local * np.cos(p_j.phi) - u_eta_local * np.sin(p_j.phi)
        vg = u_xi_local * np.sin(p_j.phi) + u_eta_local * np.cos(p_j.phi)
        
        vn_ind = ug * p_i.nx + vg * p_i.ny
        vt_ind = ug * p_i.tx + vg * p_i.ty
        
        return vn_ind, vt_ind

    def solve(self):
        N = self.n_panels
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        for i in range(N):
            
            b[i] = - self.U_inf * self.panels[i].nx
            
            for j in range(N):
                vn_induced, _ = self.get_influence(i, j)
                A[i, j] = vn_induced
                
        sigmas = np.linalg.solve(A, b)
        
        for i in range(N):
            self.panels[i].sigma = sigmas[i]
            
            Vt_total = self.U_inf * self.panels[i].tx
            
            for j in range(N):
                _, vt_induced = self.get_influence(i, j)
                Vt_total += vt_induced * sigmas[j]
            
            self.panels[i].Vt = Vt_total
            self.panels[i].Cp = 1.0 - (Vt_total / self.U_inf)**2

    def calculate_flow_field(self, X, Y):

        u = np.full_like(X, self.U_inf)
        v = np.zeros_like(X)
        
        print("Calculating Flow Field...")
        for p in self.panels:
            dx = X - p.x0
            dy = Y - p.y0
            
            xi = dx * np.cos(p.phi) + dy * np.sin(p.phi)
            eta = -dx * np.sin(p.phi) + dy * np.cos(p.phi)
            
            r1_sq = xi**2 + eta**2
            r2_sq = (xi - p.L)**2 + eta**2
            
            r1_sq = np.maximum(r1_sq, 1e-12)
            r2_sq = np.maximum(r2_sq, 1e-12)
            
            ln_term = 0.5 * np.log(r2_sq / r1_sq)
            
            theta1 = np.arctan2(eta, xi)
            theta2 = np.arctan2(eta, xi - p.L)
            d_theta = theta2 - theta1
            

            u_xi = (p.sigma / (2 * np.pi)) * ln_term
            u_eta = (p.sigma / (2 * np.pi)) * d_theta
            
            u += u_xi * np.cos(p.phi) - u_eta * np.sin(p.phi)
            v += u_xi * np.sin(p.phi) + u_eta * np.cos(p.phi)
            
        return u, v

def build_scherer_geometry(divergence_angle_deg=10, n_panels_per_wall=100):
    panels = []
    
    R = 0.15
    g = 0.04           
    L_inlet = 1.0    
    L_outlet = 1.0   
    
    theta = np.radians(divergence_angle_deg) 
    
    x_points = []
    y_points = []
    
    dx = L_inlet / 50
    for x in np.arange(-L_inlet, -R, dx):
        x_points.append(x)
        y_points.append(-(g/2 + R))
        
    n_curve = 50
    for alpha in np.linspace(np.pi, np.pi/2, n_curve):
        xc = R * np.cos(alpha)
        yc = -(g/2 + R) + R * np.sin(alpha)
        x_points.append(xc)
        y_points.append(yc)
        

    n_div = 50
    for x in np.linspace(0, L_outlet, n_div)[1:]:
        x_points.append(x)
        y_points.append(-(g/2 + x * np.tan(theta)))


    x_points = [-x for x in x_points]
    zipped_points = sorted(zip(x_points, y_points))
    x_points = [p[0] for p in zipped_points]
    y_points = [p[1] for p in zipped_points]
    
    for k in range(len(x_points)-1):
        panels.append(Panel(x_points[k], y_points[k], x_points[k+1], y_points[k+1]))

    x_top = x_points[::-1] 
    y_top = [-y for y in y_points[::-1]] 
    
    for k in range(len(x_top)-1):
        panels.append(Panel(x_top[k], y_top[k], x_top[k+1], y_top[k+1]))
        
    return panels

def thwaites_separation(panels, kinematic_viscosity=0.15):

    wall_panels = [p for p in panels if p.yc < 0]
    wall_panels.sort(key=lambda p: p.xc)
    
    s_vals = [] 
    U_vals = [] 
    x_vals = [] 
    current_s = 0.0
    
    for p in wall_panels:
        s_vals.append(current_s + p.L/2)
        U_vals.append(abs(p.Vt)) 
        x_vals.append(p.xc)
        current_s += p.L
        
    s = np.array(s_vals)
    U = np.array(U_vals)
    x = np.array(x_vals)
    
    integrand = U**5
    integral = np.zeros_like(U)
    
    for i in range(1, len(U)):
        ds = s[i] - s[i-1]
        avg_integrand = 0.5 * (integrand[i] + integrand[i-1])
        integral[i] = integral[i-1] + avg_integrand * ds
        
    theta2 = (0.45 * kinematic_viscosity * integral) / (U**6 + 1e-9)
    dUds = np.gradient(U, s)
    lambda_param = (theta2 / kinematic_viscosity) * dUds
    
    separation_index = np.where(lambda_param < -0.09)[0]
    sep_point = None
    if len(separation_index) > 0:
        post_throat_indices = [idx for idx in separation_index if x[idx] > 0]
        if post_throat_indices:
            idx = post_throat_indices[0]
            sep_point = (x[idx], s[idx])
            
    return x, s, U, lambda_param, sep_point

def mask_grid(X, Y, panels):

    bottom_panels = [p for p in panels if p.yc < 0]
    top_panels = [p for p in panels if p.yc > 0]
    
    bottom_panels.sort(key=lambda p: p.xc)
    top_panels.sort(key=lambda p: p.xc)
    
    bx = [p.xc for p in bottom_panels]
    by = [p.yc for p in bottom_panels]
    tx = [p.xc for p in top_panels]
    ty = [p.yc for p in top_panels]
    
    f_bot = interp1d(bx, by, kind='linear', fill_value="extrapolate")
    f_top = interp1d(tx, ty, kind='linear', fill_value="extrapolate")
    

    y_bot = f_bot(X)
    y_top = f_top(X)
    
    mask = (Y < y_bot) | (Y > y_top)
    return mask

panels = build_scherer_geometry(divergence_angle_deg=10)
U_inf = 100.0
solver = SourcePanelMethod(panels, U_inf=U_inf) 

solver.solve()

nu_air = 0.15 
x_bl, s_bl, U_bl, lam_bl, sep = thwaites_separation(panels, nu_air)

plt.figure(figsize=(12, 16))


plt.subplot(4, 1, 1)
plt.title("Panel Method Geometry (10-deg Divergence)")

for p in panels:
    plt.plot([p.x0, p.x1], [p.y0, p.y1], 'k-', linewidth=1)
plt.ylabel("Y (cm)")
plt.grid(True)
plt.xlim(-1.0, 1.0)

plt.subplot(4, 1, 2)





print(f"Flow Separation Predicted at x = {sep[0]:.4f} cm")

    


wall_panels = [p for p in panels if p.yc < 0]
wall_panels.sort(key=lambda p: p.xc)
xc_wall = [p.xc for p in wall_panels]
cp_wall = [p.Cp for p in wall_panels]

V_inlet = abs(wall_panels[0].Vt)


sep_x = sep[0]
idx_nearest = (np.abs(np.array(xc_wall) - sep_x)).argmin()
cp_at_sep = cp_wall[idx_nearest]

plt.plot(sep_x, cp_at_sep, 'ro', label='Separation Point', zorder=10)
plt.axvline(x=sep_x, color='r', linestyle='--', alpha=0.5)
plt.text(sep_x + 0.1, cp_at_sep, 'Separation', color='red')
plt.legend()

plt.plot(xc_wall, cp_wall, 'b-', linewidth=2, label='Panel Method Cp')
plt.gca().invert_yaxis() 
plt.ylabel("Cp")
plt.title("Wall Pressure Coefficient")
plt.xlim(-1.0, 1.0)
plt.grid(True)
plt.legend()


x_min, x_max = -1.0, 1.0
y_min, y_max = -0.8, 0.8
Nx, Ny = 80, 50
xg = np.linspace(x_min, x_max, Nx)
yg = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(xg, yg)

u_field, v_field = solver.calculate_flow_field(X, Y)
V_mag = np.sqrt(u_field**2 + v_field**2)
Cp_field = 1.0 - (V_mag / U_inf)**2

mask = mask_grid(X, Y, panels)
u_field = np.ma.array(u_field, mask=mask)
v_field = np.ma.array(v_field, mask=mask)
Cp_field = np.ma.array(Cp_field, mask=mask)

plt.subplot(4, 1, 3)
plt.title("Velocity Streamlines")
for p in panels:
    plt.plot([p.x0, p.x1], [p.y0, p.y1], 'k-', linewidth=1)
    
speed = np.sqrt(u_field**2 + v_field**2)
strm = plt.streamplot(X, Y, u_field, v_field, color=speed, cmap='plasma', density=1.5, linewidth=1)
plt.colorbar(strm.lines, label='Velocity Magnitude')
plt.axis([x_min, x_max, y_min, y_max])
plt.ylabel("Y (cm)")
plt.xlim(-1.0, 1.0)

plt.subplot(4, 1, 4)
plt.title("Pressure Coefficient Contours")
for p in panels:
    plt.plot([p.x0, p.x1], [p.y0, p.y1], 'k-', linewidth=1)
    
levels = np.linspace(np.min(Cp_field), np.max(Cp_field), 30)
cp_plot = plt.contourf(X, Y, Cp_field, levels=levels, cmap='RdBu_r', extend='both')
plt.colorbar(cp_plot, label='$C_p$')
plt.axis([x_min, x_max, y_min, y_max])
plt.xlabel("Axial Distance x (cm)")
plt.ylabel("Y (cm)")



plt.tight_layout()
plt.xlim(-1.0, 1.0)
plt.show()
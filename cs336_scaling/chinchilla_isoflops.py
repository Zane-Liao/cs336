import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1: Load data
with open("../data/isoflops_curves.json", "r") as f:
    data = json.load(f)

compute = np.array([d["compute_budget"] for d in data])
params = np.array([d["parameters"] for d in data])
loss = np.array([d["final_loss"] for d in data])

# 2: Find optimal runs per compute
C_unique = sorted(set(compute))
N_opt, L_opt = [], []

for C in C_unique:
    mask = compute == C
    best = np.argmin(loss[mask])  # loss
    N_opt.append(params[mask][best])
    L_opt.append(loss[mask][best])

N_opt = np.array(N_opt)
C_unique = np.array(C_unique)

# 3: Fit scaling law N_opt(C) = a * C^b
def power_law(C, a, b):
    return a * np.power(C, b)

popt_N, _ = curve_fit(power_law, C_unique, N_opt)
aN, bN = popt_N

# Step 4: Compute D_opt = C / (6 * N_opt)
# compute ≈ 6 * N * D
D_opt = C_unique / (6 * N_opt)
popt_D, _ = curve_fit(power_law, C_unique, D_opt)
aD, bD = popt_D

# Step 5: Extrapolate to 1e23 and 1e24 FLOPs
for target in [1e23, 1e24]:
    n_pred = power_law(target, *popt_N)
    d_pred = power_law(target, *popt_D)
    print(f"For {target:.0e} FLOPs:")
    print(f"  → Predicted optimal model size: {n_pred:.3e} parameters")
    print(f"  → Predicted optimal dataset size: {d_pred:.3e} tokens\n")

# Step 6: Plot Scaling Laws

def plot_scaling(x, y, a, b, ylabel, color):
    plt.figure(figsize=(6, 4))
    plt.loglog(x, y, "o", color=color, label="Data (Optimal points)")
    x_fit = np.logspace(np.log10(x.min()), 24, 100)
    plt.loglog(x_fit, power_law(x_fit, a, b), "-", color="black",
               label=f"Fit: y = {a:.2e} * C^{b:.3f}")
    plt.xlabel("Compute Budget (FLOPs)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Compute (IsoFLOPs Scaling)")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

# (1) N_opt(C)
plot_scaling(C_unique, N_opt, aN, bN, "Optimal Model Size N_opt(C)", "C0")

# (2) D_opt(C)
plot_scaling(C_unique, D_opt, aD, bD, "Optimal Dataset Size D_opt(C)", "C1")

print("Done!")

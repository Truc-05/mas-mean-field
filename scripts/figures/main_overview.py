"""Main overview figure: the single figure a reader should remember. Six
equal panels, every number verified, the full chain from instrument to
repairability at a glance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import style as st
import data as dt
import data_extra as de

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_overview")

fig, axes = st.grid(2, 3, figsize=(13.5, 8.0))
st.suptitle(fig, "Dependence is not enough")

# ---------------------------------------------------------------- a: instrument
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "Instrument validation: sign test")
null_mean, null_sd = dt.C3["null_mean"], dt.C3["null_sd"]
x = np.linspace(null_mean - 4 * null_sd, 66, 400)
y = np.exp(-0.5 * ((x - null_mean) / null_sd) ** 2)
y = y / y.max()
ax.fill_between(x, 0, y, color=st.CHURN, alpha=0.25, label="simulated null (summary)")
ax.plot(x, y, color=st.CHURN, lw=1.3)
ax.axvline(dt.C3["S_obs"], color=st.SHARED, lw=2.2)
ax.scatter([dt.C3["S_obs"]], [0.02], color=st.SHARED, s=70, zorder=5, marker="D")
ax.text(dt.C3["S_obs"] - 1.5, 0.75, f"observed\n{dt.C3['S_obs']} / {dt.C3['K']}",
        color=st.SHARED, fontsize=9.5, fontweight="bold", ha="right")
ax.text(null_mean, 0.95, f"null mean {null_mean:.1f}", color=st.INK_SECONDARY,
        fontsize=8.3, ha="center")
ax.set_xlabel("strata with shared $>$ independent dependence")
ax.set_ylabel("density (normalized)")
ax.set_yticks([])
ax.text(0.98, 0.92, "$p<0.0005$", transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=st.INK)

# ---------------------------------------------------------- b: noise canyon
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Raw churn is mostly noise")
regimes = ["hard", "easy"]
raw = [dt.CHURN_STRICT_HARD["dY_cross_mean"], dt.CHURN_STRICT_EASY["dY_cross_mean"]]
noise = [dt.CHURN_STRICT_HARD["C_within_mean"], dt.CHURN_STRICT_EASY["C_within_mean"]]
excess = [dt.CHURN_STRICT_HARD["dC_mean"], dt.CHURN_STRICT_EASY["dC_mean"]]
xpos = np.arange(2)
w = 0.36
ax.bar(xpos - w / 2, raw, width=w, color=st.CHURN, alpha=0.55, label="raw $\\Delta Y$")
ax.bar(xpos + w / 2, noise, width=w, color=st.INK_MUTED, alpha=0.85, label="noise floor $C_{within}$")
for i, e in enumerate(excess):
    ax.annotate(f"$\\delta C$={e:+.3f}", xy=(xpos[i], max(raw[i], noise[i]) + 0.012),
                 ha="center", fontsize=8.3, color=st.SHARED, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(["hard / low-comp.", "easy / high-comp."])
ax.set_ylabel("decision-change rate")
ax.legend(loc="upper right")

# ------------------------------------------------------- c: mechanism gap
ax = axes[0, 2]
st.style_axes(ax)
st.panel_title(ax, "c", "Copying moves decisions more than sharing")
ests = ["caliper", "NN-ATT", "nested-LPM"]
vals = [dt.MACRO["caliper_gap"], dt.MACRO["nn_att"], dt.MACRO["mech_coef"]]
cis = [dt.MACRO_CI["caliper_gap"], dt.MACRO_CI["nn_att"], dt.MACRO_CI["mech_coef"]]
ypos = np.arange(3)
for i, (v, ci) in enumerate(zip(vals, cis)):
    ax.plot(ci, [ypos[i]] * 2, color=st.COPYING, lw=2.6, solid_capstyle="round")
    ax.scatter([v], [ypos[i]], color=st.COPYING, s=60, zorder=5, edgecolor="white", linewidth=1)
    ax.text(ci[1] + 0.008, ypos[i], f"{v:.3f}", va="center", fontsize=8.6, color=st.INK, fontweight="bold")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels(ests)
ax.set_xlabel("mechanism gap: copying $-$ shared ($\\Delta Y$)")
ax.set_xlim(-0.02, 0.22)

# ---------------------------------------------------- d: accuracy equivalence
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "d", "Net accuracy stays near zero")
H2 = dt.DA_REGIME["H2"]
cats = ["hard", "easy", "pooled"]
means = [H2["hard"]["mean"], H2["easy"]["mean"], H2["all"]["mean"]]
los = [H2["hard"]["lo90"], H2["easy"]["lo90"], H2["all"]["lo90"]]
his = [H2["hard"]["hi90"], H2["easy"]["hi90"], H2["all"]["hi90"]]
colors = [st.REGIME_HARD, st.REGIME_EASY, st.GOOD]
ypos = np.arange(3)
ax.axvspan(-0.05, 0.05, color=st.GOOD_FACE, alpha=0.35, zorder=0)
for i in range(3):
    ax.plot([los[i], his[i]], [ypos[i]] * 2, color=colors[i], lw=2.6, solid_capstyle="round")
    ax.scatter([means[i]], [ypos[i]], color=colors[i], s=60, zorder=5, edgecolor="white", linewidth=1)
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels(cats)
ax.set_xlabel(r"$E[\Delta A]$, 90% CI  (shaded = $\pm$0.05 equivalence band)")
ax.set_xlim(-0.065, 0.065)

# -------------------------------------------------------- e: directional anatomy
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "e", "Where the $\\Delta Y$ gap goes")
comp = de.MECH_GAP_BREAKDOWN
labels = ["harmful\n(break)", "beneficial\n(rescue)", "churn\n(wrong$\\to$wrong)"]
vals = [comp["p_break_gap"], comp["p_rescue_gap"], comp["p_churn_gap"]]
colors = [st.CRITICAL, st.GOOD, st.CHURN]
bottoms = [0, vals[0], vals[0] + vals[1]]
for i in range(3):
    ax.bar([0], [vals[i]], bottom=[bottoms[i]], width=0.5, color=colors[i], label=labels[i].replace("\n", " "))
    ax.text(0.33, bottoms[i] + vals[i] / 2, f"{vals[i]:.3f}", va="center", fontsize=8.6,
            color=colors[i], fontweight="bold")
ax.set_xlim(-0.6, 1.3)
ax.set_xticks([0])
ax.set_xticklabels([f"total gap\n{comp['dY_gap']:.3f}"])
ax.set_ylabel(r"$\Delta Y$ gap composition (matched $D$)")
ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.0))

# -------------------------------------------------------- f: repairability identity
ax = axes[1, 2]
st.style_axes(ax, spine="none")
st.panel_title(ax, "f", "Repairability is a net, not a rate")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.text(5, 8.6, r"$H_{rep} = p_{+-} - p_{-+}$", ha="center", fontsize=15, color=st.INK, fontweight="bold")
ax.text(5, 7.0, "recoverable headroom = harmful flips $-$ beneficial flips", ha="center",
        fontsize=8.8, color=st.INK_SECONDARY)
bh, be = dt.CHURN_STRICT_HARD, dt.CHURN_STRICT_EASY
rows = [
    ("matched-$D$ (copying $-$ shared)", de.MECH_GAP_BREAKDOWN["p_break_gap"], de.MECH_GAP_BREAKDOWN["p_rescue_gap"]),
    ("breadth, hard regime", de.BREAKDOWN_HARD["p_break"], de.BREAKDOWN_HARD["p_rescue"]),
    ("breadth, easy regime", de.BREAKDOWN_EASY["p_break"], de.BREAKDOWN_EASY["p_rescue"]),
]
y0 = 5.4
for i, (lab, pb, pr) in enumerate(rows):
    y = y0 - i * 1.7
    ax.text(0.3, y + 0.55, lab, fontsize=8.3, color=st.INK_SECONDARY, va="bottom")
    ax.barh([y], [pb], left=[0], height=0.45, color=st.CRITICAL, alpha=0.85)
    ax.barh([y], [pr], left=[pb], height=0.45, color=st.GOOD, alpha=0.85)
    ax.text(pb + pr + 0.15, y, f"$H_{{rep}}$={pb-pr:+.3f}", fontsize=8.0, va="center",
            color=st.INK, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

st.savefig(fig, OUT)
print("wrote", OUT)

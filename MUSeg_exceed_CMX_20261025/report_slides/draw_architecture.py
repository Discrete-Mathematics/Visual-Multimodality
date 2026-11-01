"""SelectiveMine-F architecture as a wide, print-style method figure.

Same Matplotlib construction as before (rounded boxes, orthogonal arrows),
restyled and relaid out like a conference method figure: aligned columns,
thin gray arrows, and a small set of muted branch colors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Data coordinates match a 17.6 x 9.4 inch figure, so corners stay square.
W, H = 1760, 940
fig = plt.figure(figsize=(17.6, 9.4), dpi=120)
ax = fig.add_axes([0.012, 0.02, 0.976, 0.96])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

INK = "#1F2937"
MUTED = "#6B7280"
ARROW = "#4B5563"
LINE = 0.95

# fill, edge
RGB = ("#E7F0FA", "#1E4E86")
DEPTH = ("#F8F1E4", "#8C5A16")
GEO = ("#EEEFFA", "#3A4688")
FUSE = ("#E6F4F1", "#0F6B60")
LATENT = ("#E8F6EC", "#1B6B38")
SELECT = ("#F7F4EC", "#8A6230")
PANEL = "#F7F8FA"
PANEL_EDGE = "#E5E7EB"


def panel(x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=12",
            fc=PANEL, ec=PANEL_EDGE, lw=0.8, zorder=0,
        )
    )


def box(x, y, w, h, fill, edge, text, size=11.5, color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=6",
            fc=fill, ec=edge, lw=1.05, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=size, color=color,
        linespacing=1.35, zorder=3,
    )
    return (x, y, w, h)


def label(x, y, text, color=MUTED):
    ax.text(x, y, text, ha="left", va="center", fontsize=11, color=color, zorder=3)


def L(b):
    x, y, w, h = b
    return (x, y + h / 2)


def R(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def T(b):
    x, y, w, h = b
    return (x + w / 2, y)


def B(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def arrow(pts, color=ARROW, head=True):
    pts = np.asarray(pts, dtype=float)
    verts = [pts[0]]
    codes = [Path.MOVETO]
    if len(pts) == 2:
        verts.append(pts[1])
        codes.append(Path.LINETO)
    else:
        for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
            d1, d2 = b - a, c - b
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-6 or n2 < 1e-6:
                verts.append(b)
                codes.append(Path.LINETO)
                continue
            d1, d2 = d1 / n1, d2 / n2
            k = min(9.0, n1 / 2, n2 / 2)
            verts += [b - d1 * k, b, b + d2 * k]
            codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
        verts.append(pts[-1])
        codes.append(Path.LINETO)
    ax.add_patch(
        FancyArrowPatch(
            path=Path(verts, codes),
            arrowstyle=("-|>,head_length=6.5,head_width=3.2" if head else "-"),
            mutation_scale=1, lw=LINE, color=color,
            shrinkA=0, shrinkB=0, zorder=1,
            capstyle="round", joinstyle="round",
        )
    )


def dot(x, y):
    ax.add_patch(Circle((x, y), 3.1, fc=ARROW, ec="white", lw=0.6, zorder=4))


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
panel(28, 28, 1048, 500)
panel(1096, 28, 636, 500)
panel(28, 676, 1704, 232)

label(48, 50, "外观与可靠性融合", FUSE[1])
label(1116, 50, "几何支路", GEO[1])
label(48, 698, "选择与对数混合", SELECT[1])

# Appearance lane
rgb = box(52, 74, 168, 58, *RGB, "RGB")
mit = box(268, 74, 248, 58, *RGB, "MiT-B2 RGB 编码器", 11)
app = box(564, 74, 292, 58, *FUSE, "外观解码器\n构造时复制融合解码器", 10)
zr = box(904, 74, 140, 58, *LATENT, "$z_R$", 15)

# Depth lane and gate
depth = box(52, 186, 168, 54, *DEPTH, "打包深度")
dep = box(268, 186, 248, 54, *DEPTH, "DepthEncoder")
gate = box(564, 168, 480, 92, *FUSE,
           "四层 ReliabilityGate\n$F = F_{rgb} + \\alpha \\cdot \\delta(F_{depth})$", 11.5)
valid = box(52, 318, 168, 48, *DEPTH, "有效掩码", 11)
luma = box(268, 318, 248, 48, *RGB, "亮度", 11)
fuse = box(564, 408, 292, 54, *FUSE, "融合解码器  FuseDecoder", 11)
zb = box(904, 408, 140, 54, *LATENT, "$z_B$", 15)

# Geometry tower, right panel. Five stages on a vertical rhythm.
hha = box(1234, 78, 360, 54, *GEO, "四通道几何")
stem = box(1234, 164, 360, 54, *GEO, "1×1 stem，4 → 3")
genc = box(1234, 250, 360, 54, *GEO, "MiT-B2 几何编码器")
gdec = box(1234, 336, 360, 54, *FUSE, "几何解码器")
zg = box(1314, 422, 200, 54, *LATENT, "$z_G$", 15)

# Merge
present = box(52, 572, 250, 62, *SELECT, "s：整幅图是否有深度", 11.5)
zf = box(360, 564, 1088, 78, *LATENT,
         "$z_F = z_B + \\tanh(\\mathrm{open})\\cdot \\mathrm{mix}\\cdot s\\cdot z_G$", 13)

# Selection row
evidence = box(52, 748, 430, 86, *SELECT,
               "52 通道证据，先 detach\n两个概率、绝对差、两个熵、掩码、亮度", 11)
router = box(530, 760, 300, 64, *SELECT, "路由器\n1×1，深度卷积 3×3，1×1", 11)
g = box(872, 766, 150, 52, *LATENT, "$g = \\sigma(a)$", 14)
mix = box(1064, 748, 390, 86, *SELECT,
          "s = 1\n$\\log\\left((1-g)\\,p_R + g\\,p_F\\right)$", 12)
out = box(1496, 748, 210, 86, *LATENT, "s = 0 时改为 $\\log p_R$\n返回 logits", 11.5)

# ---------------------------------------------------------------------------
# Wires. Gutters only; no arrow crosses a box.
# ---------------------------------------------------------------------------
arrow([R(rgb), L(mit)])
arrow([R(mit), L(app)])
arrow([R(app), L(zr)])

# RGB encoder also enters the gate, through the gap above the depth row.
arrow([R(mit), (546, 103), (546, 158), (804, 158), T(gate)])
dot(546, 103)

arrow([R(depth), L(dep)])
arrow([R(dep), L(gate)])

# Mask and luma rise into the bottom of the gate through the open band.
arrow([T(valid), (136, 296), (700, 296), (700, 260)])
arrow([T(luma), (392, 296)], head=False)
dot(392, 296)

arrow([B(gate), (804, 408)])
arrow([R(fuse), L(zb)])

arrow([B(hha), T(stem)])
arrow([B(stem), T(genc)])
arrow([B(genc), T(gdec)])
arrow([B(gdec), T(zg)])

# Latents into z_F. z_B drops in the gap under the fusion decoder.
arrow([B(zb), (974, 564)])
arrow([B(zg), (1414, 564)])
arrow([(136, 366), (136, 572)])
arrow([R(present), L(zf)])

# z_R takes the channel above the appearance lane, then the outer-left margin.
arrow([T(zr), (974, 66), (24, 66), (24, 791), L(evidence)])
arrow([L(valid), (24, 342)], head=False)
dot(24, 342)
arrow([B(luma), (392, 384), (24, 384)], head=False)
dot(24, 384)

arrow([R(evidence), L(router)])
arrow([R(router), L(g)])
arrow([R(g), L(mix)])
arrow([R(mix), L(out)])

# z_F enters the mixture from above; s joins along the bottom margin.
arrow([(1259, 642), (1259, 748)])
arrow([B(present), (177, 656), (24, 656), (24, 888), (1259, 888), (1259, 834)])
dot(24, 656)

out_path = r"d:\Downloads\MUSeg_exceed_CMX_20261025\report_slides\selectivemine_architecture"
fig.savefig(out_path + ".png", dpi=220, facecolor="white")
fig.savefig(out_path + ".pdf", facecolor="white")
print(out_path)

#!/usr/bin/env python3
"""
BSG軸（backspin, sidespin, gyro）と速度ベクトル・水平面の幾何学関係を図示
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Yu Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def get_bsg_axes(theta_deg, phi_deg):
    """BSG の 3 軸の単位ベクトルを返す"""
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)
    # ジャイロ軸 = 速度方向（Nathan Excel: uz = +sinθ）
    g = np.array([cth * sph, -cth * cph, sth])
    # バックスピン軸（Nathan Excel: b = [-cosφ, sinφ, 0]）
    b = np.array([-cph, sph, 0])
    # サイドスピン軸（Nathan Excel: s = [-sinθsinφ, -sinθcosφ, cosθ]）
    s = np.array([-sth * sph, -sth * cph, cth])
    return b, s, g


def main():
    theta_deg, phi_deg = -5.0, 15.0
    b, s, g = get_bsg_axes(theta_deg, phi_deg)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scale = 1.2
    origin = np.zeros(3)

    # 速度ベクトル（ジャイロ軸 = g）
    ax.quiver(0, 0, 0, g[0], g[1], g[2], color="C0", arrow_length_ratio=0.15, lw=2.5, label="ĝ（ジャイロ軸 = 速度方向）")

    # バックスピン軸 b（水平面内）
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color="C1", arrow_length_ratio=0.15, lw=2.5, label="b̂（バックスピン軸・水平）")

    # サイドスピン軸 s
    ax.quiver(0, 0, 0, s[0], s[1], s[2], color="C2", arrow_length_ratio=0.15, lw=2.5, label="ŝ（サイドスピン軸）")

    # 水平面（Z=0）のグリッド
    xx = np.linspace(-0.6, 0.6, 5)
    yy = np.linspace(-0.6, 0.6, 5)
    Xh, Yh = np.meshgrid(xx, yy)
    Zh = np.zeros_like(Xh)
    ax.plot_surface(Xh, Yh, Zh, alpha=0.15, color="gray")
    ax.plot([-0.5, 0.5], [0, 0], [0, 0], "k--", alpha=0.4, lw=1)
    ax.plot([0, 0], [-0.5, 0.5], [0, 0], "k--", alpha=0.4, lw=1)

    # 速度の水平成分（点線）
    g_h = np.array([g[0], g[1], 0])
    if np.linalg.norm(g_h) > 0.05:
        g_h_n = g_h / np.linalg.norm(g_h) * 0.5
        ax.quiver(0, 0, 0, g_h_n[0], g_h_n[1], 0, color="C0", arrow_length_ratio=0.1, ls="--", alpha=0.7)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z（上向き正）")
    ax.set_title(f"BSG 軸と速度・水平面の関係\n（θ={theta_deg}°, φ={phi_deg}°）")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_box_aspect([1, 1, 1])

    # 注釈
    ax.text2D(0.02, 0.02, "・ĝ ∥ 速度、b̂ は水平面内で ĝ の水平成分に垂直\n・ŝ は鉛直成分を持つ", transform=ax.transAxes, fontsize=9, va="bottom")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "bsg_axes_figure.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

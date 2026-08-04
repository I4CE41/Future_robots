<div align="center">

# 🤖 Future Robots
### *Energy-Aware NMPC for Autonomous Mobile Manipulators*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CasADi](https://img.shields.io/badge/CasADi-IPOPT-orange?style=for-the-badge)](https://web.casadi.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-Physics_Sim-blue?style=for-the-badge)](https://pybullet.org)
[![Paper](https://img.shields.io/badge/Paper-CISTEM_2026-green?style=for-the-badge)](#-research-paper)

<p align="center">
  <b>A simulation research framework combining cell-resolved LiFePO₄ battery modeling with real-time energy-aware Nonlinear Model Predictive Control.</b>
</p>

---

</div>

## 📌 Overview

This codebase accompanies the peer-reviewed research paper:

> **"A BYD-Inspired Cell-to-Pack Energy Model and Energy-Aware NMPC for Safe Autonomous Mobile Manipulators"**  
> *A. Louizini, A. Garmat, K. Guesmi — CISTEM 2026 (v3)*

### 🔍 Key Research Question
> *Does cell-resolved battery modeling combined with energy-aware NMPC produce more reliable, longer-endurance autonomous missions than simpler energy abstractions?*

> [!NOTE]
> This framework simulates contact-based collisions and high-fidelity battery state estimation using PyBullet and IPOPT numerical optimization.

---

## ✨ Main Features

| Feature | Description |
| :--- | :--- |
| **🔋 Cell-to-Pack Battery Model** | BYD-inspired 8S LFP pack ($25.6\text{ V}$, $20\text{ Ah}$, $512\text{ Wh}$) with gated EKF SoC correction, passive balancing, and motor waste-heat recovery. |
| **⚡ Energy-Aware NMPC** | 7-state nonlinear MPC solved via **CasADi / IPOPT** with dynamic SoC-dependent weighting $\phi(z)$. |
| **🗺️ Hybrid Navigation Pipeline** | Clearance-aware **Global A\*** + reactive **Pure-Pursuit / CBF** local avoidance with DWA / MPPI support. |
| **🧱 Corridor Memory** | Local costmap with perpendicular sampling, corridor hysteresis, and persistent JSON-backed cross-run memory. |
| **📊 Real Physics Testing** | Contact-based collision detection using PyBullet physics simulation. |

---

## ⚡ Controller Variants

Five controller variants are evaluated in the paper:

| Variant | Energy Weight | SoC Throttle | Regen | Cell EKF | Balancing | Dispersion | Speed Scale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`full`** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `1.0` |
| **`pack_level`** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `1.0` |
| **`no_energy`** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | `1.0` |
| **`no_regen`** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | `1.0` |
| **`speed_only`** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | `1.4` |

> [!WARNING]
> Running the **`full`** controller variant increases CPU computational load during real-time IPOPT solving due to multi-cell matrix evaluations.

---

## 📋 Roadmap & Tasks

- [x] Cell-to-Pack LiFePO₄ Battery Model integration
- [x] Energy-Aware NMPC via CasADi / IPOPT
- [x] PyBullet physics simulation environment
- [ ] Hardware-in-the-loop (HIL) physical testing
- [ ] Real-time ROS 2 node migration

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone [https://github.com/I4CE41/Future_robots.git](https://github.com/I4CE41/Future_robots.git)
cd Future_robots

# Install dependencies
pip install -r requirements.txt
python main.py 

python main20.py

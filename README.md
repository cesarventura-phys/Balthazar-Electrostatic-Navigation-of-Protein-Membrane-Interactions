# Balthazar-Electrostatic-Navigation-of-Protein-Membrane-Interactions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![APBS](https://img.shields.io/badge/APBS-3.0%2B-orange)](https://apbs.poissonboltzmann.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Overview

**Balthazar** is a modular computational platform designed to systematically explore the electrostatic energy landscape of protein-membrane interactions. By solving the linearized Poisson-Boltzmann equation (LPBE) through the APBS library and coupling it with geometric manipulation algorithms, Balthazar identifies physically relevant initial configurations for molecular dynamics simulations at a fraction of the computational cost.

**Key insight:** Mean-field electrostatics is sufficient to predict privileged initial orientations that can serve as optimal starting points for molecular dynamics simulations.

### 🎯 Core Features

- **Systematic rotational scanning** using Euler angles (θ, φ, ψ) parameterization
- **Automatic membrane orientation** via inertia tensor alignment
- **Numerical noise suppression** (De-noiser module) to eliminate grid discretization artifacts
- **Energy-distance profiling** (Approacher module) for binding curve analysis
- **Modular architecture** with specialized tools for different tasks
- **Complete workflow automation** from PQR preparation to heatmap visualization

## 🏗️ Architecture

Balthazar consists of five integrated tools:

┌─────────────────┐
│ Rotator │
│ (Steric contact │
│ detection) │
└────────┬────────┘
│ R_min distance
▼
┌─────────────────────────────────────────────────┐
│ Balthazar │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│ │ Orientation │─▶│ APBS │─▶│De-noiser│ │
│ │ Scanning │ │ Solver │ │ │ │
│ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────┘
│ │
▼ ▼
┌───────────────┐ ┌───────────────┐
│ Approacher │ │ ALite │
│(Energy-distance│ │(Instantaneous │
│ profiling) │ │ config viewer)│
└───────────────┘ └───────────────┘


## 🔬 Methodology

### Theoretical Foundation

The platform solves the **linearized Poisson-Boltzmann equation**:

$$\nabla \cdot [\epsilon(\mathbf{r}) \nabla V(\mathbf{r})] = -\rho_{fixed}(\mathbf{r}) + \kappa^2 \epsilon_{sol} V(\mathbf{r})$$

where:
- $\epsilon(\mathbf{r})$ is the position-dependent dielectric constant
- $\rho_{fixed}$ represents fixed atomic charges
- $\kappa^{-1}$ is the Debye length (∼8 Å under physiological conditions)

### Geometric Processing

1. **Centering**: Molecules are centered at their geometric centers (not center of mass)
2. **Membrane alignment**: Principal axes of inertia align the membrane in the XY plane
3. **Euler rotations**: Protein orientations are parameterized using three consecutive rotations (Z-Y'-Z'' convention)

### De-noiser Module

A critical innovation addressing numerical noise from grid discretization:

$$U_{binding} = U_{CS2} - (U_{prot} + U_{mem})$$

where $U_{CS2}$ is the combined system energy, and $U_{prot}$, $U_{mem}$ are isolated molecule energies calculated in identical grid positions. This subtraction cancels grid-dependent artifacts.

## 📊 Validation Studies

### Model Systems
The platform was validated on increasing complexity:

| System | Expected Behavior | Result |
|--------|------------------|--------|
| Point charges | Coulombic 1/r dependence | ✓ Confirmed |
| Dipole vs. charge | ∝ cosθ angular dependence | ✓ Confirmed |
| Dipole vs. membrane | Complex angular landscape | ✓ Captured |
| Quadrupole | 2-fold symmetry | ✓ Captured |
| Octupole | 4-fold symmetry | ✓ Captured |

### Biological Systems
25 unique protein-membrane combinations were analyzed:

**Proteins:**
- 1IFB (Fatty acid-binding protein)
- 2JU3 (Membrane-binding domain)
- 2XV9 (Peripheral membrane protein)
- Ov-FAR-1 (Fatty acid and retinol-binding protein)
- Sj-FABPc (Schistosoma fatty acid-binding protein)

**Membrane compositions (POPC/SOPS mixtures):**
- 0:1 (fully charged, SOPS only)
- 1:0 (neutral, POPC only)
- 1:1 (mixed)
- 1:3 (charged-rich)
- 3:1 (neutral-rich)

## 📈 Key Findings

1. **Defined Energy Landscapes**: All systems showed well-defined local extrema, confirming that electrostatics alone guides initial orientation.
2. **Lipid Composition Dependence**: High charge density yields interactions orders of magnitude larger; neutral membranes qualitatively invert the landscape.
3. **Euler Angle Dependence**: The third Euler angle contributes minimal variation, justifying 2D scanning as computationally optimal.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- APBS 3.0+
- NumPy, SciPy, Matplotlib
- PyMOL (optional, for visualization)

### Installation
git clone https://github.com/yourusername/balthazar.git
cd balthazar
pip install -r requirements.txt
\


Basic Workflow

# 1. Check steric contact with Rotator
python rotator.py --protein protein.pqr --membrane membrane.pqr --output configs/

# 2. Run main energy scan with Balthazar
python balthazar.py \
    --protein protein.pqr \
    --membrane membrane.pqr \
    --theta-range 0 180 15 \
    --phi-range 0 360 15 \
    --distance 55 \
    --output results/

# 3. Profile optimal orientations with Approacher
python approacher.py \
    --protein protein.pqr \
    --membrane membrane.pqr \
    --theta 45 --phi 120 \
    --distance-range 50 160 20 \
    --output distance_profile/



Input File Format
Balthazar requires PQR files (PDB format with charge and radius columns):
ATOM      1  N   ARG A   1      10.521  15.745  11.345  0.320  1.85
ATOM      2  CA  ARG A   1      11.123  14.635  10.615  0.180  1.87
...



Example Results
Energy Heatmap (2JU3 protein on charged membrane)

https://docs/images/2ju3_C0S1.png
*X-axis: φ angle (0-360°), Y-axis: θ angle (0-180°). Violet indicates favorable (low energy) configurations.*
Distance Profiles

https://docs/images/T_2ju3_C0S1.png
Blue: Most favorable orientation, Orange: Least favorable orientation. Note the electrostatic minimum before steric contact.

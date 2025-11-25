# NOVA: Neural Architecture Discovery with Physics for Out-of-Distribution and Versatile Adaptation

NOVA is a physics-guided neural architecture search (NAS) framework designed to discover
neural physics solvers that generalize robustly under distribution shifts and enable
stable long-time predictions across PDE systems. This repository contains the
implementation for the Navier–Stokes (NS) experiments under Dirichlet boundary conditions.

---

## 🌟 Highlights

- **Physics-guided NAS**  
  Searches for architectures whose inductive bias aligns with PDE structure.

- **Data-driven training + physics-guided evaluation**  
  Use supervised training first, then adapt with PDE residuals.

- **Zero-shot OOD generalization**  
  Test on unseen geometries or flow conditions *without retraining*.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Typical dependencies:

- torch  
- numpy  
- matplotlib  
- scipy  

---

## 📁 Project Structure

```
NOVA/
  models/                # Neural architectures (MLP, U-Net, NOVA models)
  search/
    NS/
      train_DNN_NS_DirichletBC.py
      train_PINN_NS_DirichletBC_pseudo_inverse.py
  util/                  # PDE operators, losses, schedulers, utilities
  misc/                  # FLOPs computation tools
  .gitignore
```

---

## 🚀 Training (Supervised Navier–Stokes)

We first train a **data-driven neural network (DNN)** on Navier–Stokes data.

### Script
```
search/NS/train_DNN_NS_DirichletBC.py
```

### Run
```bash
python search/NS/train_DNN_NS_DirichletBC.py
```

### Description
- Loads NS dataset  
- Builds selected neural architecture  
- Trains using supervised velocity/pressure regression  
- Enforces Dirichlet boundary conditions  
- Saves model checkpoint + architecture specification  

This yields an efficient feature representation for NS flow fields.

---

## 🧪 Testing / Zero-shot Physics Adaptation (PINN Pseudo-inverse)

After supervised learning, NOVA evaluates the trained architecture with a **physics-informed pseudo-inverse update**.

### Script
```
search/NS/train_PINN_NS_DirichletBC_pseudo_inverse.py
```

### Run
```bash
python search/NS/train_PINN_NS_DirichletBC_pseudo_inverse.py
```

### Description
- Loads trained model and architecture  
- Freezes feature extractor  
- Computes NS PDE residuals (momentum + continuity)  
- Solves a **closed-form pseudo-inverse** for final-layer weights  
- Produces physics-consistent predictions for ID + OOD cases  

This achieves **zero-shot generalization** without gradient-based retraining.

---

## 📊 Outputs

Scripts generate:

- Training curves  
- Flow field visualizations (u, v, p)  
- RMSE for ID and OOD evaluation  
- FLOPs / parameter count  

---

## 📖 Citation

If you use NOVA for research, please cite:

```
@article{zhao2025nova,
  title={Neural Architecture Discovery with Physics for Out-of-Distribution and Versatile Adaptation},
  author={Zhao, Wei and others},
  year={2025}
}
```

---

## 📩 Contact

Dr. Wei Zhao  
Centre for Frontier AI Research (CFAR)
Agency for Science, Technology and Research (A*STAR)
Singapore
Email: weiz@a-star.edu.sg


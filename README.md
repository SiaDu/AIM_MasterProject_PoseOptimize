# AIM_MasterProject_PoseOptimize

**One‑liner:** Rule‑based pose optimizer for clearer single‑frame **gesture**, **silhouette**, and **camera‑facing**.

---

## Project Introduction

This is a lightweight tool to improve a **single 3D character pose** under a given camera. The system first **scores** the pose (Gesture line, Silhouette clarity, Camera‑facing), then applies **small, safe joint adjustments** using simple **Line‑of‑Action rules** or a tiny **genetic search** (DEAP).

* **Input:** one pose (joints JSON) + optional camera config.
* **Output:** an optimized pose (JSON) and a before/after preview image.
* **Why:** clearer storytelling poses without changing style; fast, deterministic, **no dataset required**.

---

## 🧰 Environment

* Python: **3.10.12**
* Pinned deps:

  * numpy==2.2.6
  * matplotlib==3.10.3
  * transforms3d==0.4.2
  * deap==1.4.3

**Install (pip):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy==2.2.6 matplotlib==3.10.3 transforms3d==0.4.2 deap==1.4.3
```

**requirements.txt**

```txt
numpy==2.2.6
matplotlib==3.10.3
transforms3d==0.4.2
deap==1.4.3
```

---

## 🚀 How to Run

**Notebook (end‑to‑end):**

```bash
jupyter notebook PoseOptimize_GAmodel.ipynb
```

**Scripts:**

```bash
python process_pose.py     # preprocess pose
python LoA_optimize.py     # optimize (LoA / GA)
python Visualize.py        # save before/after preview
python score.py            # print readability score
```

> Scripts are self‑contained. Check file headers for optional arguments.

---

## 🧠 Method (brief)

We compute three scores (gesture continuity, silhouette clarity, camera‑facing). Then we apply **rule‑based tweaks** (e.g., open shoulders, elbow pop‑out, head tilt, spine arc) with joint limits. Optionally, a small **GA** (DEAP) searches within safe ranges. The best pose and a **before/after** image are saved to `outputs/`.

---

## 📌 Features

* ✔ Score: **gesture / silhouette / facing**
* ✔ Optimize: **rule‑based** + optional **GA** (DEAP)
* ✔ Simple **before/after** visualization
* ✔ Configurable via **YAML**; reproducible outputs
* ✔ Works offline; **no dataset required**

---

## 👤 Author

Du Siyao (SiaDu)
MSc Artificial Intelligence for Media, Bournemouth University
Contact: [s5722875@bournemouth.ac.uk](mailto:s5722875@bournemouth.ac.uk)
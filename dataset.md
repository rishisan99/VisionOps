# Dataset Documentation

## Dataset Name

**PlantVillage Dataset (Tomato Subset)**

---

## Source

The dataset used in this project is derived from the **PlantVillage Dataset**, originally created by researchers at **Pennsylvania State University** and collaborators.

Official source:

-   https://github.com/spMohanty/PlantVillage-Dataset

This project uses **only a subset** of the original dataset, focusing on tomato leaf images.

---

## Classes Used

The following classes were selected to balance realism, interpretability, and computational feasibility:

| Class Name                | Images          |
| ------------------------- | --------------- |
| Tomato_healthy            | 1000            |
| Tomato_Early_blight       | 1000            |
| Tomato_Late_blight        | 1000            |
| Tomato_Septoria_leaf_spot | 1000            |
| **Total**                 | **4000 images** |

---

## Dataset Structure (Original)

```

PlantVillage/
├── Tomato_healthy/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
└── Tomato_Septoria_leaf_spot/

```

Images are RGB photographs of tomato leaves captured under varying conditions.

---

## Dataset License

The PlantVillage dataset is released under the:

**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**

License details:

-   https://creativecommons.org/licenses/by-sa/4.0/

### What this means:

-   ✅ You may use, share, and adapt the dataset
-   ✅ You may use it for commercial and non-commercial purposes
-   ⚠️ Attribution is required
-   ⚠️ Derivative works must use the same license

This repository **does not redistribute the dataset**.
Users are expected to download the dataset directly from the original source.

---

## Attribution

If you use this dataset, please cite the original authors:

> Hughes, D. P., & Salathé, M. (2015).
> An open access repository of images on plant health to enable the development of mobile disease diagnostics.
> _arXiv preprint arXiv:1511.08060._

---

## Usage in This Project

-   Raw dataset stored locally under:
    `data/_source/PlantVillage/` (excluded from version control)
-   All preprocessing outputs are versioned using **DVC**
-   No raw images are committed to GitHub
-   Dataset lineage is tracked through pipeline artifacts

---

## Ethical & Practical Considerations

-   The model is intended for **educational and demonstration purposes**
-   Predictions should not be considered a substitute for professional agricultural diagnosis
-   Dataset bias may exist due to controlled capture environments

---

## Notes

This dataset was chosen because:

-   It is widely recognized in computer vision research
-   Visual disease patterns are interpretable
-   It supports explainability techniques like Grad-CAM

---

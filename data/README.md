# Data

Dataset and derived files for the COMP47950 QML project.

- **`raw/`** — Downloaded or original datasets (reserved for future use).
- **`processed/`** — Train/test splits, scaled features, or other prepared data (reserved for future use).

**Current setup:** Iris, Wine, and Breast Cancer Wisconsin are loaded on-the-fly from scikit-learn via the `qml_project` package (e.g. `qml_project.datasets`). No local caching; splits are reproducible via a fixed `random_state` (42). See the main notebook for split sizes matching [20] Table 1.

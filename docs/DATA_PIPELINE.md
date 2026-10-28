# Smart Education — Data Pipeline & Dataset Architecture

## 1. Overview & Pedagogical Data Foundation

The Smart Education platform is grounded in real learner interaction telemetry from the **EdNet-KT3** benchmark dataset, tracking multi-turn question attempts, lecture views, and response latencies across thousands of curriculum concepts.

---

## 2. Preprocessed Artifacts Inventory

All authentic preprocessed data artifacts are located in `data/processed/` and verified against Git history:

| Artifact Path | Format | Size | Description | Active Status |
| :--- | :---: | :---: | :--- | :---: |
| `data/processed/metadata.pkl` | Pickle | 129.5 KB | Global vocabulary mappings: `question_to_idx` (11,555), `tag_to_idx` (187), `part_to_idx` (7), normalization statistics | Verified & Loaded |
| `data/processed/questions_contents.csv` | CSV | 554.3 KB | Catalog of 13,169 practice questions with concept tag IDs and curriculum parts | Verified & Loaded |
| `data/processed/lectures_contents.csv` | CSV | 25.5 KB | Catalog of 1,021 remedial video lectures with duration (seconds) and target tags | Verified & Loaded |
| `data/processed/sample_test_students.csv` | CSV | 1.63 MB | Interaction event sequences for authentic held-out test students (e.g., student `u100` with 22 events) | Verified & Loaded |
| `data/processed/split_users.json` | JSON | 4.45 MB | Strict user-level partition manifest defining train, validation, and test cohort user sets | Verified & Loaded |
| `data/processed/leakage_report.txt` | Text | 974 B | Audit confirmation verifying strict zero-overlap user partitions | Verified & Loaded |

---

## 3. Interaction Feature Formulation

Each interaction event is structured into a 9-dimensional tensor for sequential neural processing:

1. `question_id`: Global question index mapped via `question_to_idx` (1 to 11,555).
2. `part`: TOEIC/curriculum section identifier (Parts 1 through 7).
3. `tag1`, `tag2`, `tag3`: Up to 3 concept tags associated with the pedagogical item.
4. `is_correct`: Binary student outcome (1 = Correct, 0 = Incorrect).
5. `response_time`: Normalized duration spent before answer submission.
6. `source_encoded`: Interaction source origin (e.g., diagnostic assessment, practice drill).
7. `platform_encoded`: Client platform origin (Mobile App, Web Browser).

---

## 4. Sequence Formulation & Target Alignment

- **Sequence Window:** Standard sequence length of $T = 50$ interactions.
- **Target Formulation:** Next-event prediction: the model predicts whether interaction $t+1$ will be answered correctly given historical interactions $1 \dots t$.
- **Masking:** Padded positions are masked with Boolean tensor masks so padding never contaminates gradients or state updates.

---

## 5. Offline Data Pipeline Policy

> [!IMPORTANT]
> The runtime frontend dashboard and inference services **never execute full-dataset preprocessing, data downloads, or retraining**.
> All data loading is strictly read-only and cached in memory using `@st.cache_data`.

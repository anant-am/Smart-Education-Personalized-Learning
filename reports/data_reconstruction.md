# Data Reconstruction & Event Synthesis Documentation
**Project:** AI-Based Personalized Learning Recommendation System – Smart Education  
**Dataset:** EdNet-KT3 (Hierarchical Multi-Action Student Interaction Logs)  
**Date:** September 12, 2026  

---

## 1. Objective and Problem Formulation

In the EdNet-KT3 hierarchy, student interactions are recorded as discrete, granular action events rather than flat question-attempt rows. A student solving a question does not produce a single log line with a correctness flag; instead, the interaction unfolds over multiple sequential events:
- `enter`: The learner opens an item bundle (e.g., question bundle `b...`, lecture video `l...`, or explanation document `e...`).
- `respond`: The learner inputs or changes their answer choice for a specific question `q...`.
- `submit`: The learner finalizes and commits their answers for the bundle `b...`.
- `quit`: The learner closes a lecture or explanation session.

The objective of data reconstruction is to transform these raw, low-level event streams into structured, chronologically ordered learning events with ground-truth correctness labels, precise response times, and behavioral context while preventing future data leakage.

---

## 2. Event Reconstruction State Machine

Each user log file (`u{id}.csv`) is sorted chronologically by `timestamp` ($t$) and processed through a deterministic state machine:

```text
       [Raw Stream]
            │
            ▼
      action_type?
      ├── enter(b*)  ───► Open new Question Bundle (record bundle_enter_ts)
      ├── respond(q*)───► Append candidate (question_id, user_answer, respond_ts)
      ├── submit(b*) ───► Finalize Question Attempt:
      │                   - Match final user_answer to Contents correct_answer
      │                   - Compute response_time = final_respond_ts - bundle_enter_ts
      │                   - Emit verified Question Event
      ├── enter(l*)  ───► Open Lecture Session (record lecture_enter_ts)
      ├── quit(l*)   ───► Close Lecture Session (duration = quit_ts - enter_ts)
      ├── enter(e*)  ───► Open Explanation Session (record explanation_enter_ts)
      └── quit(e*)   ───► Close Explanation Session (duration = quit_ts - enter_ts)
```

### 2.1 Question Attempt Resolution
- **Multi-Response Resolution:** If a learner changes their answer multiple times within a single bundle, the final response preceding the `submit` event represents the committed submission. Earlier responses reflect deliberation behavior (`num_responses` feature).
- **Correctness Determination:** 
  $$y = \mathbb{I}[\text{user\_answer} = \text{correct\_answer}]$$
  The ground-truth `correct_answer` is retrieved strictly via an inner join with `questions.csv`.
- **Response Time Calculation:**
  $$\Delta t_{\text{response}} = t_{\text{final\_respond}} - t_{\text{bundle\_enter}}$$
  Any negative or corrupt timestamps are clamped to zero. Outliers exceeding the 99th percentile are clipped during standardization.

### 2.2 Lecture and Explanation Engagement
- **Duration Tracking:** Session engagement is computed as $t_{\text{quit}} - t_{\text{enter}}$.
- **Inactive Session Capping:** Durations exceeding 3,600,000 ms (1 hour) are capped to prevent background browser tab artifacts from distorting engagement representations.

---

## 3. Sequential Feature Derivation (Zero Future Leakage)

To equip sequence models (LSTM, Transformer, BERT-NCF, CNN-LSTM) and tabular estimators with rich learner representations, sequential performance indicators are derived strictly using past events:

1. **Cumulative Prior Accuracy ($\text{prev\_acc}_t$):**
   $$\text{prev\_acc}_t = \frac{1}{t-1} \sum_{k=1}^{t-1} y_k \quad (\text{for } t > 1; \quad 0 \text{ for } t=1)$$
2. **Recent Window Accuracy ($\text{rec5\_acc}_t$):**
   $$\text{rec5\_acc}_t = \frac{1}{\min(5, t-1)} \sum_{k=\max(1, t-5)}^{t-1} y_k$$
3. **Cumulative Attempt Count ($\text{count}_t$):**
   $$\text{count}_t = t - 1$$
4. **Inter-Event Time Interval ($\Delta t_{\text{prev}}$):**
   $$\Delta t_{\text{prev}} = t_{\text{enter}, t} - t_{\text{enter}, t-1} \quad (\text{for } t > 1; \quad 0 \text{ for } t=1)$$

All lag features strictly exclude the current target $y_t$ and any future events $t' > t$, satisfying causality constraints.

---

## 4. Cohort Partitioning & Transformation Strategy

- **Development Cohort (`KT-3 (1000)`):** 1,000 students. All feature scalers ($\mu_{\text{rt}}, \sigma_{\text{rt}}, \mu_{\text{tsp}}, \sigma_{\text{tsp}}$) and categorical vocabularies (`question_to_idx`, `tag_to_idx`, `platform_map`, `source_map`) are fitted exclusively on this cohort.
- **Unseen Test Cohort (`KT-3 (250) TEST`):** 250 students. Transformed using frozen parameters from the development cohort. Any unobserved questions or tags are mapped to index `0` (padding/unknown).

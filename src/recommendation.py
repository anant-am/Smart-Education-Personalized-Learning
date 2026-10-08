"""
Smart Education Project — Personalized Learning Resource Recommendation Engine
=============================================================================
Provides:
1. RecommendationEngine: Maps diagnosed student learning gaps to authentic EdNet
   learning materials (lectures, explanations, practice questions).
2. Remedial Intervention Hierarchy: Prioritizes lectures > explanations > targeted practice.
3. Candidate Exclusion: Filters out materials already attempted by the student.
4. Concept Dependency Graph: Lightweight graph mapping prerequisite and co-occurring
   concepts from curriculum metadata (Future-Work Item 3 / literature direction).
5. Academic Ranking Metrics: Precision@K, Recall@K, NDCG@K, HitRate@K.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set


class RecommendationEngine:
    """
    Intervention engine recommending authentic EdNet learning resources
    tailored to student concept deficiencies.
    """
    def __init__(self, questions_df: pd.DataFrame, lectures_df: pd.DataFrame):
        self.questions_df = questions_df.copy() if not questions_df.empty else pd.DataFrame()
        self.lectures_df = lectures_df.copy() if not lectures_df.empty else pd.DataFrame()

        # Build concept dependency / co-occurrence graph
        self.concept_graph = self._build_concept_graph()

    def _build_concept_graph(self) -> Dict[int, Set[int]]:
        """
        Builds a concept co-occurrence graph from multi-tagged questions.
        If a student struggles on concept A, related co-occurring concepts B in the
        same TOEIC question bundles can be considered.
        """
        graph = {}
        if self.questions_df.empty or 'tags' not in self.questions_df.columns:
            return graph

        for tags_val in self.questions_df['tags'].dropna():
            tags = [int(t.strip()) for t in str(tags_val).split(';') if t.strip().isdigit()]
            for t1 in tags:
                if t1 not in graph:
                    graph[t1] = set()
                for t2 in tags:
                    if t1 != t2:
                        graph[t1].add(t2)
        return graph

    def get_resources_for_concept(self, concept_id: int,
                                  resource_types: List[str] = ['lecture', 'explanation', 'question']) -> List[Dict[str, Any]]:
        """
        Finds authentic EdNet resources addressing a target concept ID.
        """
        resources = []
        concept_str = str(concept_id)

        # 1. Video Lectures
        if 'lecture' in resource_types and not self.lectures_df.empty:
            rel_lectures = self.lectures_df[self.lectures_df['tags'].astype(str) == concept_str]
            for _, row in rel_lectures.iterrows():
                resources.append({
                    'type': 'lecture',
                    'id': str(row['lecture_id']),
                    'concept': concept_id,
                    'part': row.get('part', None),
                    'duration_sec': int(row.get('video_length', 0)) // 1000 if pd.notna(row.get('video_length')) else 0
                })

        # 2. Explanations and Questions
        if ('explanation' in resource_types or 'question' in resource_types) and not self.questions_df.empty:
            def matches_concept(tv):
                if pd.isna(tv):
                    return False
                return concept_str in [t.strip() for t in str(tv).split(';')]

            rel_q = self.questions_df[self.questions_df['tags'].apply(matches_concept)]

            for _, row in rel_q.iterrows():
                qid = str(row['question_id'])
                exp_id = str(row.get('explanation_id')) if pd.notna(row.get('explanation_id')) else None

                if 'explanation' in resource_types and exp_id and exp_id != 'nan':
                    resources.append({
                        'type': 'explanation',
                        'id': exp_id,
                        'concept': concept_id,
                        'linked_question': qid,
                        'part': row.get('part', None)
                    })

                if 'question' in resource_types:
                    resources.append({
                        'type': 'question',
                        'id': qid,
                        'concept': concept_id,
                        'part': row.get('part', None),
                        'bundle_id': str(row.get('bundle_id', ''))
                    })

        return resources

    def recommend_for_gaps(self, learning_gaps: List[Dict[str, Any]],
                           top_k: int = 5,
                           exclude_attempted: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Generates top-K recommendations targeted at weak concepts.
        Priority: lectures (conceptual foundation) > explanations (worked solutions) > questions (active practice).
        """
        exclude_attempted = exclude_attempted or set()
        recommendations = []
        seen_resource_ids = set()

        type_priority = {'lecture': 1, 'explanation': 2, 'question': 3}

        for gap in learning_gaps:
            cid = int(gap.get('concept_id', gap.get('concept', 0)))
            resources = self.get_resources_for_concept(cid)

            # Sort candidate materials by pedagogy priority
            resources.sort(key=lambda r: type_priority.get(r['type'], 99))

            for res in resources:
                rid = res['id']
                if rid in exclude_attempted or rid in seen_resource_ids:
                    continue

                recommendations.append(res)
                seen_resource_ids.add(rid)

                if len(recommendations) >= top_k:
                    return recommendations

        # If more materials needed, expand using concept dependency graph
        if len(recommendations) < top_k and learning_gaps:
            for gap in learning_gaps:
                cid = int(gap.get('concept_id', gap.get('concept', 0)))
                related_concepts = self.concept_graph.get(cid, set())
                for rel_c in related_concepts:
                    rel_resources = self.get_resources_for_concept(rel_c)
                    rel_resources.sort(key=lambda r: type_priority.get(r['type'], 99))
                    for res in rel_resources:
                        rid = res['id']
                        if rid in exclude_attempted or rid in seen_resource_ids:
                            continue
                        recommendations.append(res)
                        seen_resource_ids.add(rid)
                        if len(recommendations) >= top_k:
                            return recommendations

        return recommendations[:top_k]

    def evaluate_recommendations(self, student_recommendations: Dict[str, List[str]],
                                 student_future_interactions: Dict[str, List[str]],
                                 k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Computes formal ranking evaluation metrics (Precision@K, Recall@K, NDCG@K, HitRate@K).
        """
        metrics = {}
        for k in k_values:
            p_list, r_list, ndcg_list, hit_list = [], [], [], []

            for uid, recs in student_recommendations.items():
                actual = student_future_interactions.get(uid, [])
                if not actual:
                    continue

                top_k = recs[:k]
                hits = set(top_k).intersection(set(actual))
                hit_count = len(hits)

                p = hit_count / k if k > 0 else 0.0
                r = hit_count / len(actual) if len(actual) > 0 else 0.0
                hit = 1.0 if hit_count > 0 else 0.0

                dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(top_k) if item in actual])
                idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(k, len(actual)))])
                ndcg = dcg / idcg if idcg > 0 else 0.0

                p_list.append(p)
                r_list.append(r)
                ndcg_list.append(ndcg)
                hit_list.append(hit)

            metrics[f"Precision@{k}"] = float(np.mean(p_list)) if p_list else 0.0
            metrics[f"Recall@{k}"] = float(np.mean(r_list)) if r_list else 0.0
            metrics[f"NDCG@{k}"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0
            metrics[f"HitRate@{k}"] = float(np.mean(hit_list)) if hit_list else 0.0

        return metrics

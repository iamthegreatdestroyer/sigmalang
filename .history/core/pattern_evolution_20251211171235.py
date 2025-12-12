"""
Pattern Evolution Layer - Phase 2A.4

Implements pattern clustering, abstraction, and emergent pattern discovery.

Key Components:
- PatternClusterer: Agglomerative clustering with silhouette scoring
- PatternAbstractor: Longest common subsequence extraction for abstractions
- EmergentPatternDiscoverer: KL divergence-based novelty detection
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

logger = logging.getLogger(__name__)


def _compute_jaccard_distance_lower(str1_lower: str, str2_lower: str) -> float:
    """Compute Jaccard distance between two already-lowercased strings."""
    set1 = set(str1_lower.split())
    set2 = set(str2_lower.split())

    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    similarity = intersection / union if union > 0 else 0.0
    return 1.0 - similarity


def _distance_task(args: Tuple[int, int, str, str]) -> Tuple[int, int, float]:
    i, j, str_i, str_j = args
    return i, j, _compute_jaccard_distance_lower(str_i, str_j)


# ============================================================================
# PATTERN CLUSTERING
# ============================================================================

@dataclass
class ClusterResult:
    """Result of pattern clustering operation."""
    cluster_id: int
    patterns: List[str]
    centroid: Optional[np.ndarray] = None
    size: int = 0
    cohesion: float = 0.0  # Average intra-cluster distance
    separation: float = 0.0  # Average inter-cluster distance
    silhouette_score: float = 0.0

    def __post_init__(self):
        """Initialize size."""
        self.size = len(self.patterns)


class PatternClusterer:
    """
    Clusters patterns using agglomerative (hierarchical) clustering.
    
    Features:
    - Configurable distance metrics
    - Automatic optimal cluster detection via silhouette scoring
    - Hierarchical dendrogram generation
    """

    def __init__(self, distance_metric: str = "euclidean"):
        """
        Initialize clusterer.

        Args:
            distance_metric: Distance metric for clustering (euclidean, cosine, etc.)
        """
        self.distance_metric = distance_metric
        self.linkage_matrix = None
        self.distance_matrix = None

    def compute_pattern_distance(
        self,
        pattern1: Any,
        pattern2: Any
    ) -> float:
        """
        Compute distance between two patterns.

        Args:
            pattern1: First pattern
            pattern2: Second pattern

        Returns:
            Distance value (0 = identical, 1 = maximally different)
        """
        # Convert patterns to strings for comparison
        str1 = str(pattern1).lower()
        str2 = str(pattern2).lower()

        # Compute Jaccard similarity then convert to distance
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        similarity = intersection / union if union > 0 else 0.0
        distance = 1.0 - similarity

        return distance

    def cluster_patterns(
        self,
        pattern_dict: Dict[str, Any],
        num_clusters: Optional[int] = None,
        distance_threshold: float = 0.5,
        n_jobs: int = 1,
        backend: str = "process",
    ) -> List[ClusterResult]:
        """
        Cluster patterns using hierarchical clustering.

        Args:
            pattern_dict: Dict of pattern_id -> pattern object
            num_clusters: Target number of clusters (auto-detect if None)
            distance_threshold: Distance threshold for cluster formation

        Returns:
            List of ClusterResult objects
        """
        if len(pattern_dict) < 2:
            # Single pattern or empty
            results = []
            for cluster_id, (pid, pattern) in enumerate(pattern_dict.items()):
                results.append(ClusterResult(
                    cluster_id=cluster_id,
                    patterns=[pid],
                    cohesion=0.0,
                    separation=0.0,
                    silhouette_score=1.0
                ))
            return results

        # Convert patterns to IDs list
        pattern_ids = list(pattern_dict.keys())
        patterns = [pattern_dict[pid] for pid in pattern_ids]

        # Compute pairwise distance matrix
        n_patterns = len(patterns)
        distance_matrix = np.zeros((n_patterns, n_patterns))

        if n_jobs is None or n_jobs <= 1 or n_patterns < 3:
            for i in range(n_patterns):
                for j in range(i + 1, n_patterns):
                    dist = self.compute_pattern_distance(patterns[i], patterns[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        else:
            backend_normalized = (backend or "process").strip().lower()
            if backend_normalized not in {"process", "thread"}:
                raise ValueError(f"Invalid backend: {backend}. Expected 'process' or 'thread'.")

            # Pre-stringify once to reduce per-task overhead.
            pattern_strs = [str(p).lower() for p in patterns]

            worker_count = max(1, min(int(n_jobs), os.cpu_count() or 1))
            tasks = (
                (i, j, pattern_strs[i], pattern_strs[j])
                for i in range(n_patterns)
                for j in range(i + 1, n_patterns)
            )

            # Process pools can fail in constrained environments; fall back to threads.
            try:
                if backend_normalized == "process":
                    with ProcessPoolExecutor(max_workers=worker_count) as ex:
                        for i, j, dist in ex.map(_distance_task, tasks, chunksize=64):
                            distance_matrix[i, j] = dist
                            distance_matrix[j, i] = dist
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as ex:
                        for i, j, dist in ex.map(_distance_task, tasks):
                            distance_matrix[i, j] = dist
                            distance_matrix[j, i] = dist
            except Exception as e:
                logger.warning(
                    "Parallel distance computation failed (%s); falling back to sequential.",
                    e,
                )
                for i in range(n_patterns):
                    for j in range(i + 1, n_patterns):
                        dist = self.compute_pattern_distance(patterns[i], patterns[j])
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist

        self.distance_matrix = distance_matrix

        # Convert to condensed distance matrix for scipy
        condensed_distances = squareform(distance_matrix)

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(
            condensed_distances,
            method='average'  # Average linkage
        )

        # Determine optimal number of clusters
        if num_clusters is None:
            num_clusters = self._find_optimal_clusters(distance_threshold)

        num_clusters = max(1, min(num_clusters, len(patterns)))

        # Get cluster assignments
        cluster_labels = fcluster(
            self.linkage_matrix,
            num_clusters,
            criterion='maxclust'
        )

        # Build cluster results
        clusters: Dict[int, List[str]] = {}
        for pattern_id, label in zip(pattern_ids, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pattern_id)

        # Compute cluster metrics
        results = []
        for cluster_id, (label, pids) in enumerate(clusters.items()):
            cohesion = self._compute_cohesion(pids, distance_matrix, pattern_ids)
            separation = self._compute_separation(
                pids,
                pattern_ids,
                distance_matrix,
                cluster_labels,
                label
            )
            silhouette = self._compute_silhouette_score(
                pids,
                pattern_ids,
                distance_matrix,
                cluster_labels,
                label
            )

            results.append(ClusterResult(
                cluster_id=cluster_id,
                patterns=pids,
                cohesion=cohesion,
                separation=separation,
                silhouette_score=silhouette
            ))

        results.sort(key=lambda c: c.size, reverse=True)
        return results

    def _find_optimal_clusters(self, distance_threshold: float) -> int:
        """
        Find optimal number of clusters using distance threshold.

        Args:
            distance_threshold: Threshold for cluster merging

        Returns:
            Optimal number of clusters
        """
        if self.linkage_matrix is None:
            return 2

        # Find where distance exceeds threshold
        distances = self.linkage_matrix[:, 2]
        num_clusters = np.sum(distances < distance_threshold) + 1

        return max(1, min(num_clusters, len(distances)))

    def _compute_cohesion(
        self,
        cluster_pids: List[str],
        distance_matrix: np.ndarray,
        all_pids: List[str]
    ) -> float:
        """Compute average intra-cluster distance."""
        if len(cluster_pids) <= 1:
            return 0.0

        # Get indices of cluster patterns
        indices = [all_pids.index(pid) for pid in cluster_pids]

        # Compute average pairwise distance within cluster
        total_dist = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_dist += distance_matrix[indices[i], indices[j]]
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _compute_separation(
        self,
        cluster_pids: List[str],
        all_pids: List[str],
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_label: int
    ) -> float:
        """Compute average inter-cluster distance."""
        # Get indices outside this cluster
        cluster_indices = set(all_pids.index(pid) for pid in cluster_pids)
        external_indices = [
            i for i, label in enumerate(cluster_labels)
            if i not in cluster_indices
        ]

        if not external_indices:
            return 0.0

        # Compute average distance to external points
        cluster_idx_list = list(cluster_indices)
        total_dist = 0.0
        count = 0

        for ci in cluster_idx_list:
            for ei in external_indices:
                total_dist += distance_matrix[ci, ei]
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _compute_silhouette_score(
        self,
        cluster_pids: List[str],
        all_pids: List[str],
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_label: int
    ) -> float:
        """
        Compute silhouette score for cluster (average -1 to 1).

        Args:
            cluster_pids: Pattern IDs in this cluster
            all_pids: All pattern IDs
            distance_matrix: Full distance matrix
            cluster_labels: Cluster assignment for all patterns
            cluster_label: This cluster's label

        Returns:
            Silhouette score (-1 to 1)
        """
        if len(cluster_pids) <= 1:
            return 1.0

        cluster_indices = [all_pids.index(pid) for pid in cluster_pids]
        other_indices = [
            i for i, label in enumerate(cluster_labels)
            if label != cluster_label
        ]

        if not other_indices:
            return 1.0

        scores = []
        for ci in cluster_indices:
            # a: average distance to other points in cluster
            a_values = [
                distance_matrix[ci, oi]
                for oi in cluster_indices
                if oi != ci
            ]
            a = np.mean(a_values) if a_values else 0.0

            # b: minimum average distance to points in other clusters
            b_values = [
                distance_matrix[ci, oi]
                for oi in other_indices
            ]
            b = np.min([np.mean(b_values)]) if b_values else 0.0

            # Silhouette: (b - a) / max(a, b)
            max_val = max(a, b)
            silhouette = (b - a) / max_val if max_val > 0 else 0.0
            scores.append(silhouette)

        return np.mean(scores)


# ============================================================================
# PATTERN ABSTRACTION
# ============================================================================

class PatternAbstractor:
    """
    Extracts abstract patterns using longest common subsequence (LCS).
    
    Features:
    - LCS extraction for finding common patterns
    - Parameterizable subroutine identification
    - Template generation from pattern sets
    """

    @staticmethod
    def extract_common_pattern(
        patterns: List[str]
    ) -> str:
        """
        Extract common pattern from list of patterns using LCS.

        Args:
            patterns: List of pattern strings

        Returns:
            Common pattern string
        """
        if not patterns:
            return ""

        if len(patterns) == 1:
            return patterns[0]

        # Start with first pattern and find LCS with each subsequent pattern
        common = patterns[0]

        for pattern in patterns[1:]:
            common = PatternAbstractor.lcs(common, pattern)
            if not common:
                break

        return common

    @staticmethod
    def lcs(s1: str, s2: str) -> str:
        """
        Compute longest common subsequence.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Longest common subsequence
        """
        m, n = len(s1), len(s2)
        
        # Build DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Reconstruct LCS
        lcs_str = []
        i, j = m, n

        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs_str.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return ''.join(reversed(lcs_str))

    @staticmethod
    def extract_parameters(
        patterns: List[str],
        template: str
    ) -> List[Dict[str, str]]:
        """
        Extract parameters from patterns given a template.

        Args:
            patterns: List of concrete patterns
            template: Abstract pattern template

        Returns:
            List of parameter dictionaries
        """
        parameters = []

        for pattern in patterns:
            params = PatternAbstractor._extract_params(pattern, template)
            parameters.append(params)

        return parameters

    @staticmethod
    def _extract_params(pattern: str, template: str) -> Dict[str, str]:
        """Extract parameters from single pattern using template."""
        params = {}

        # Simple parameter extraction - find differences between pattern and template
        pattern_parts = pattern.split()
        template_parts = template.split()

        param_idx = 0
        for ppart, tpart in zip(pattern_parts, template_parts):
            if ppart != tpart:
                param_idx += 1
                params[f"param_{param_idx}"] = ppart

        return params


# ============================================================================
# EMERGENT PATTERN DISCOVERY
# ============================================================================

@dataclass
class EmergentPattern:
    """Represents a discovered emergent pattern."""
    pattern_id: str
    pattern: Any
    novelty_score: float  # 0-1, higher = more novel
    utility_score: float  # 0-1, higher = more useful
    emergence_score: float  # 0-1, combined score
    related_patterns: List[str] = field(default_factory=list)
    emergence_reason: str = ""


class EmergentPatternDiscoverer:
    """
    Discovers emergent patterns using novelty and utility metrics.
    
    Features:
    - KL divergence-based novelty detection
    - Utility scoring based on application frequency
    - Automated pattern discovery from clusters
    """

    def __init__(self, novelty_threshold: float = 0.7):
        """
        Initialize discoverer.

        Args:
            novelty_threshold: Minimum novelty score for emergence
        """
        self.novelty_threshold = novelty_threshold
        self.pattern_frequency = {}

    def discover_patterns(
        self,
        pattern_dict: Dict[str, Any],
        clusters: List['ClusterResult'],
        frequency_data: Optional[Dict[str, int]] = None
    ) -> List[EmergentPattern]:
        """
        Discover emergent patterns from clusters.

        Args:
            pattern_dict: Dict of pattern_id -> pattern
            clusters: List of cluster results
            frequency_data: Usage frequency of each pattern

        Returns:
            List of emergent patterns
        """
        emergent = []
        self.pattern_frequency = frequency_data or {}

        # Analyze each cluster for emergent patterns
        for cluster in clusters:
            if len(cluster.patterns) < 2:
                continue

            # Get patterns in this cluster
            cluster_patterns = [
                pattern_dict.get(pid)
                for pid in cluster.patterns
                if pid in pattern_dict
            ]

            if not cluster_patterns:
                continue

            # Compute novelty: pattern dissimilarity
            novelty = self._compute_cluster_novelty(cluster_patterns)

            # Compute utility: frequency and cohesion
            utility = self._compute_cluster_utility(
                cluster.patterns,
                cluster.silhouette_score
            )

            emergence_score = 0.6 * novelty + 0.4 * utility

            if emergence_score >= self.novelty_threshold:
                # Find abstract pattern
                abstract = self._extract_abstract(cluster_patterns)

                emergent_pattern = EmergentPattern(
                    pattern_id=f"emergent_{len(emergent)}",
                    pattern=abstract,
                    novelty_score=novelty,
                    utility_score=utility,
                    emergence_score=emergence_score,
                    related_patterns=cluster.patterns,
                    emergence_reason=self._get_emergence_reason(
                        novelty, utility
                    )
                )
                emergent.append(emergent_pattern)

        # Sort by emergence score
        emergent.sort(key=lambda p: p.emergence_score, reverse=True)
        return emergent

    def _compute_cluster_novelty(self, patterns: List[Any]) -> float:
        """
        Compute novelty using KL divergence.

        Args:
            patterns: List of patterns in cluster

        Returns:
            Novelty score (0-1)
        """
        if len(patterns) < 2:
            return 0.5

        # Extract terms from all patterns
        all_terms: List[List[str]] = []
        for pattern in patterns:
            terms = str(pattern).lower().split()
            all_terms.append(terms)

        # Compute term frequency
        term_counts = Counter()
        for terms in all_terms:
            term_counts.update(terms)

        # Build probability distribution for each pattern
        vocab = set(term_counts.keys())
        vocab_size = len(vocab)

        if vocab_size == 0:
            return 0.0

        # Compute KL divergence from uniform distribution
        kl_divergences = []

        for terms in all_terms:
            # Create probability distribution
            p = np.zeros(vocab_size)
            for i, term in enumerate(sorted(vocab)):
                p[i] = (terms.count(term) + 1) / (len(terms) + vocab_size)

            # Uniform distribution
            q = np.ones(vocab_size) / vocab_size

            # KL divergence
            kl = np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))
            kl_divergences.append(min(kl / np.log(vocab_size), 1.0))  # Normalize

        novelty = np.mean(kl_divergences)
        return min(novelty, 1.0)

    def _compute_cluster_utility(
        self,
        pattern_ids: List[str],
        cohesion: float
    ) -> float:
        """
        Compute utility based on frequency and cohesion.

        Args:
            pattern_ids: Patterns in cluster
            cohesion: Cluster cohesion score

        Returns:
            Utility score (0-1)
        """
        # Frequency utility
        frequencies = [
            self.pattern_frequency.get(pid, 1)
            for pid in pattern_ids
        ]

        avg_frequency = np.mean(frequencies) if frequencies else 1.0
        # Normalize by max expected frequency (100)
        frequency_utility = min(avg_frequency / 100, 1.0)

        # Cohesion utility (tightly clustered = more useful)
        cohesion_utility = 1.0 - cohesion  # Inverse of distance

        # Combined utility
        utility = 0.5 * frequency_utility + 0.5 * cohesion_utility

        return min(utility, 1.0)

    def _extract_abstract(self, patterns: List[Any]) -> str:
        """Extract abstract pattern from concrete patterns."""
        patterns_str = [str(p) for p in patterns]
        return PatternAbstractor.extract_common_pattern(patterns_str)

    def _get_emergence_reason(self, novelty: float, utility: float) -> str:
        """Get human-readable reason for emergence."""
        reasons = []

        if novelty > 0.8:
            reasons.append("High novelty")

        if utility > 0.8:
            reasons.append("High utility")

        if novelty > 0.7 and utility > 0.7:
            reasons.append("Balanced properties")

        return "; ".join(reasons) if reasons else "Emergent properties"

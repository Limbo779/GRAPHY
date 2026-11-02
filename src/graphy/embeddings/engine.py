"""Technical Embedding Engine for Math, Physics, and Computer Science chunks & queries."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Literal, Optional

from graphy.llm.cache import LLMCache
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.math_normalizer import (
    LATEX_COMMAND_CONCEPTS,
    extract_code_identifiers,
    extract_formulas,
    extract_technical_terms_and_symbols,
    normalize_math_unicode,
)

EmbeddingMode = Literal["llama_vulkan", "technical_hash", "hybrid"]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "all", "when", "where", "who",
    "will", "can", "each", "which", "their", "there", "been", "if", "then",
    "into", "than", "its", "also", "some", "such", "only", "other", "more",
}


class TechnicalEmbeddingEngine:
    """Generates math-, physics-, figure-, and code-aware L2-normalized embeddings."""

    def __init__(
        self,
        mode: EmbeddingMode = "hybrid",
        model_manager: Optional[ModelLifecycleManager] = None,
        cache: Optional[LLMCache] = None,
        technical_dim: int = 512,
    ) -> None:
        self.mode = mode
        self.model_manager = model_manager
        self.cache = cache or LLMCache()
        self.technical_dim = technical_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of enriched chunk texts into unit-length dense vectors."""
        if not texts:
            return []

        if self.mode == "technical_hash":
            return [self._compute_technical_vector(t, self.technical_dim) for t in texts]

        if self.mode == "llama_vulkan":
            return self._embed_via_llama(texts)

        # Hybrid mode: if model_manager is provided and uses llama_server, fuse neural + technical subspace
        if self.model_manager is not None and self.model_manager.backend == "llama_server":
            neural_vecs = self._embed_via_llama(texts)
            tech_vecs = [self._compute_technical_vector(t, 256) for t in texts]
            fused: List[List[float]] = []
            for nv, tv in zip(neural_vecs, tech_vecs):
                # Weight 80% neural semantics + 20% exact math/code/symbol subspace
                combined = [0.8 * x for x in nv] + [0.2 * y for y in tv]
                norm = math.sqrt(sum(c * c for c in combined)) or 1.0
                fused.append([c / norm for c in combined])
            return fused

        return [self._compute_technical_vector(t, self.technical_dim) for t in texts]

    def embed_query(self, query_text: str) -> List[float]:
        """Enrich a problem statement or concept query with math/code descriptors and embed it."""
        norm_q = normalize_math_unicode(query_text)
        formulas = extract_formulas(norm_q)
        terms = extract_technical_terms_and_symbols(norm_q, formulas)
        code_idents = extract_code_identifiers(norm_q)

        enriched_parts = [norm_q]
        if terms:
            enriched_parts.append("Concepts: " + ", ".join(terms))
        if code_idents:
            enriched_parts.append("Code: " + ", ".join(code_idents))
        if formulas:
            enriched_parts.append("Equations: " + " ; ".join(formulas))

        enriched_query = "\n".join(enriched_parts)
        return self.embed_documents([enriched_query])[0]

    def _embed_via_llama(self, texts: List[str]) -> List[List[float]]:
        if self.model_manager is None:
            raise RuntimeError("ModelLifecycleManager is required for 'llama_vulkan' embedding mode.")

        model_id = self.model_manager.model_id_for_role("embedding")
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, txt in enumerate(texts):
            cached_vec = self.cache.get_embedding(model_id, txt)
            if cached_vec is not None:
                results[idx] = cached_vec
            else:
                missing_indices.append(idx)
                missing_texts.append(txt)

        if missing_texts:
            computed = self.model_manager.embed_texts(missing_texts)
            for idx, txt, vec in zip(missing_indices, missing_texts, computed):
                self.cache.put_embedding(model_id, txt, vec)
                results[idx] = vec

        return [r for r in results if r is not None]

    @staticmethod
    def _compute_technical_vector(text: str, dim: int = 512) -> List[float]:
        """Compute a signed-hash scientific feature vector tuned for LaTeX math, physics, and CS code."""
        norm_txt = normalize_math_unicode(text)
        lower_txt = norm_txt.lower()

        feature_weights: Dict[str, float] = {}

        # 1. Word unigrams & bigrams (scientific prose)
        words = [
            w
            for w in re.findall(r"[a-z][a-z0-9_]{1,30}", lower_txt)
            if w not in STOPWORDS
        ]
        word_counts: Dict[str, int] = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
            # Stem-like 5-char prefix for morphological invariance (e.g. "eigenvalue" / "eigenvalues")
            if len(w) >= 6:
                stem = f"stem:{w[:6]}"
                word_counts[stem] = word_counts.get(stem, 0) + 1

        for w, cnt in word_counts.items():
            weight = 1.0 + math.log(cnt)
            if w.startswith("stem:"):
                weight *= 0.85
            feature_weights[f"w:{w}"] = weight

        for i in range(len(words) - 1):
            bg = f"bg:{words[i]}_{words[i + 1]}"
            feature_weights[bg] = feature_weights.get(bg, 0.0) + 1.15

        # 2. LaTeX commands & mathematical structure tokens (boosted 2.5x)
        latex_cmds = re.findall(r"\\[A-Za-z]+", norm_txt)
        for cmd in latex_cmds:
            feature_weights[f"tex:{cmd}"] = feature_weights.get(f"tex:{cmd}", 0.0) + 2.5
            if cmd in LATEX_COMMAND_CONCEPTS:
                for concept_word in LATEX_COMMAND_CONCEPTS[cmd].split():
                    feature_weights[f"w:{concept_word.lower()}"] = (
                        feature_weights.get(f"w:{concept_word.lower()}", 0.0) + 1.8
                    )

        # 3. Mathematical expressions & sub/superscript patterns
        formulas = extract_formulas(norm_txt)
        for form in formulas:
            # Normalize whitespace and extract structural math bigrams
            clean_f = re.sub(r"\s+", "", form)
            feature_weights[f"eq:{clean_f[:40]}"] = 3.0
            math_syms = re.findall(r"\\[A-Za-z]+|[A-Za-z0-9]+|[\^_=\+\-\*/<>]", clean_f)
            for j in range(len(math_syms) - 1):
                m_tok = f"mbg:{math_syms[j]}_{math_syms[j + 1]}"
                feature_weights[m_tok] = feature_weights.get(m_tok, 0.0) + 1.6

        # 4. Code & Algorithm identifiers (boosted 2.2x)
        code_idents = extract_code_identifiers(norm_txt)
        for ident in code_idents:
            feature_weights[f"code:{ident.lower()}"] = 2.5

        # Project weighted features into dense `dim`-dimensional vector via signed SHA-256 hashing
        vec = [0.0] * dim
        for feat, weight in feature_weights.items():
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            # Project each feature onto 2 orthogonal coordinates to minimize hash collisions
            idx1 = int.from_bytes(digest[0:4], "little") % dim
            sign1 = 1.0 if (digest[4] & 1) == 0 else -1.0
            idx2 = int.from_bytes(digest[8:12], "little") % dim
            sign2 = 1.0 if (digest[12] & 1) == 0 else -1.0
            vec[idx1] += sign1 * weight
            vec[idx2] += sign2 * (weight * 0.5)

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 7) for v in vec]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute dot-product cosine similarity between two L2-normalized vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(-1.0, min(1.0, float(dot)))

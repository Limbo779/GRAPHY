"""Mathematical glyph normalizer, LaTeX formula analyzer, and technical/CS term extractor."""

from __future__ import annotations

import re
from typing import Dict, List

# Unicode mathematical symbols & Greek characters commonly emitted by PDF text extractors
UNICODE_TO_LATEX_MAP: Dict[str, str] = {
    "∀": r"\forall ",
    "∃": r"\exists ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "⊃": r"\supset ",
    "⊇": r"\supseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∅": r"\emptyset ",
    "∞": r"\infty ",
    "∂": r"\partial ",
    "∇": r"\nabla ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "∫": r"\int ",
    "∮": r"\oint ",
    "√": r"\sqrt{}",
    "≈": r"\approx ",
    "≠": r"\neq ",
    "≡": r"\equiv ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≪": r"\ll ",
    "≫": r"\gg ",
    "±": r"\pm ",
    "∓": r"\mp ",
    "×": r"\times ",
    "⋅": r"\cdot ",
    "·": r"\cdot ",
    "⊗": r"\otimes ",
    "⊕": r"\oplus ",
    "⊥": r"\perp ",
    "∥": r"\parallel ",
    "→": r"\to ",
    "↦": r"\mapsto ",
    "⇒": r"\Rightarrow ",
    "⇔": r"\Leftrightarrow ",
    "⟨": r"\langle ",
    "⟩": r"\rangle ",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}",
    "ℚ": r"\mathbb{Q}",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "⁰": "^{0}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁴": "^{4}",
    "⁵": "^{5}",
    "⁶": "^{6}",
    "⁷": "^{7}",
    "⁸": "^{8}",
    "⁹": "^{9}",
    "ⁿ": "^{n}",
    "₀": "_{0}",
    "₁": "_{1}",
    "₂": "_{2}",
    "₃": "_{3}",
    "₄": "_{4}",
    "₅": "_{5}",
    "₆": "_{6}",
    "₇": "_{7}",
    "₈": "_{8}",
    "₉": "_{9}",
    "ᵢ": "_{i}",
    "ⱼ": "_{j}",
    "ₖ": "_{k}",
}

# Canonical semantic descriptors for LaTeX commands so embeddings understand math & physics notation
LATEX_COMMAND_CONCEPTS: Dict[str, str] = {
    r"\nabla": "gradient divergence curl del vector calculus operator",
    r"\partial": "partial derivative differential equation",
    r"\int": "integral integration antiderivative calculus",
    r"\oint": "contour surface line closed integral",
    r"\sum": "summation series discrete sum",
    r"\prod": "product series",
    r"\lim": "limit convergence analysis",
    r"\det": "determinant matrix linear algebra",
    r"\ker": "kernel nullspace linear map",
    r"\dim": "dimension basis vector space",
    r"\lambda": "eigenvalue wavelength parameter",
    r"\psi": "wavefunction quantum state",
    r"\hbar": "reduced planck constant quantum mechanics",
    r"\mathbb{R}": "real numbers Euclidean space",
    r"\mathbb{C}": "complex numbers complex field",
    r"\mathcal{O}": "asymptotic time space complexity big-O",
    r"\Theta": "asymptotic tight bound complexity big-Theta",
    r"\Omega": "asymptotic lower bound sample space",
    r"\forall": "universal quantifier for all logic",
    r"\exists": "existential quantifier there exists logic",
    r"\subseteq": "subset set inclusion",
    r"\otimes": "tensor product kronecker product",
    r"\langle": "inner product bra-ket expectation value",
    r"\begin{bmatrix}": "matrix linear system transformation",
    r"\begin{pmatrix}": "matrix vector linear transformation",
    r"\frac": "rational fraction ratio derivative",
    r"\sqrt": "square root radical norm",
}

DISPLAY_MATH_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})",
    re.DOTALL,
)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)")
CODE_FENCE_RE = re.compile(
    r"(```[\w+-]*\n.*?```|\\begin\{(?:lstlisting|verbatim|algorithm|algorithmic)\}.*?\\end\{(?:lstlisting|verbatim|algorithm|algorithmic)\})",
    re.DOTALL,
)
CODE_IDENT_RE = re.compile(
    r"\b(?:def|class|function|void|int|float|double|bool|struct|template|algorithm|Procedure)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
TECHNICAL_TERM_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:'s)?\s+(?:Theorem|Lemma|Law|Equation|Inequality|Identity|Principle|Algorithm|Rule|Method|Operator|Matrix|Space|Transform|Distribution|Function|Series|Integral|Group|Ring|Field))\b"
)
BOLD_DEFINED_TERM_RE = re.compile(r"\*\*([A-Za-z0-9_\-\s]{3,45})\*\*|\\textbf\{([A-Za-z0-9_\-\s]{3,45})\}")


def normalize_math_unicode(text: str) -> str:
    """Convert raw PDF Unicode mathematical glyphs into standard LaTeX commands."""
    out = text
    # Replace ligatures first
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "--",
    }
    for lig, rep in ligatures.items():
        if lig in out:
            out = out.replace(lig, rep)

    # Convert isolated unicode math lines or inline expressions
    for uni_char, latex_cmd in UNICODE_TO_LATEX_MAP.items():
        if uni_char in out:
            out = out.replace(uni_char, latex_cmd)

    # Convert LaTeX \[ ... \] to $$ ... $$ and \( ... \) to $ ... $ for uniform Markdown+KaTeX rendering
    out = re.sub(r"\\\[(.*?)\\\]", lambda m: f"$$\n{m.group(1).strip()}\n$$", out, flags=re.DOTALL)
    out = re.sub(r"\\\((.*?)\\\)", lambda m: f"${m.group(1).strip()}$", out, flags=re.DOTALL)
    return out


def extract_formulas(text_md: str) -> List[str]:
    """Extract all display and inline mathematical expressions from a chunk."""
    formulas: List[str] = []
    for m in DISPLAY_MATH_RE.finditer(text_md):
        f = re.sub(r"\s+", " ", m.group(1)).strip()
        if f and f not in formulas:
            formulas.append(f)
    for m in INLINE_MATH_RE.finditer(text_md):
        f = m.group(1).strip()
        if len(f) >= 2 and f not in formulas:
            formulas.append(f)
    return formulas[:25]


def extract_code_identifiers(text_md: str) -> List[str]:
    """Extract function, class, and algorithm names from code blocks and prose."""
    idents: List[str] = []
    for m in CODE_IDENT_RE.finditer(text_md):
        ident = m.group(1).strip()
        if ident not in idents:
            idents.append(ident)
    return idents[:20]


def extract_technical_terms_and_symbols(text_md: str, formulas: List[str]) -> List[str]:
    """Extract named theorems/laws, bolded defined terms, and semantic math descriptors."""
    terms: List[str] = []

    for m in TECHNICAL_TERM_RE.finditer(text_md):
        t = m.group(1).strip()
        if t not in terms:
            terms.append(t)

    for m in BOLD_DEFINED_TERM_RE.finditer(text_md):
        t = (m.group(1) or m.group(2) or "").strip()
        if t and t not in terms:
            terms.append(t)

    combined_math = " ".join(formulas) + " " + text_md
    for cmd, desc in LATEX_COMMAND_CONCEPTS.items():
        if cmd in combined_math and desc not in terms:
            terms.append(desc)

    return terms[:30]


def estimate_tokens(text: str) -> int:
    """Accurate token estimator for technical prose + LaTeX + code (approx 1.35 tokens/word + symbols)."""
    words = len(re.findall(r"\w+", text))
    math_ops = len(re.findall(r"[\\{}$^_=+\-*/<>()\[\]]", text))
    return max(1, int(words * 1.3 + math_ops * 0.35))

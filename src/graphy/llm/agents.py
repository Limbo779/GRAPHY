"""Single-Responsibility Micro-Agents and Pydantic Schemas for GRAPHY & Companion."""

from __future__ import annotations

from pathlib import Path
from typing import Generic, List, Literal, Optional, Type, TypeVar
from pydantic import BaseModel, Field

from graphy.llm.model_manager import ModelRole

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


# =============================================================================
# 1. ProblemBoundaryAgent (GRAPHY Companion)
# =============================================================================
class ProblemBoundaryInput(BaseModel):
    raw_block_text: str
    source_name: str = "external_problem_set"


class ExtractedProblemItem(BaseModel):
    original_number: str
    statement_md: str
    subparts: List[str] = Field(default_factory=list)
    quarantined_hint: Optional[str] = None
    quarantined_solution: Optional[str] = None


class ProblemBoundaryOutput(BaseModel):
    problems: List[ExtractedProblemItem]


# =============================================================================
# 2. PrerequisiteJudgeAgent (Graph Construction)
# =============================================================================
class PrerequisitePairInput(BaseModel):
    chunk_a_id: str
    chunk_a_breadcrumb: str
    chunk_a_text: str
    chunk_b_id: str
    chunk_b_breadcrumb: str
    chunk_b_text: str


class PrerequisiteDecision(BaseModel):
    direction: Literal["A_PREREQ_OF_B", "B_PREREQ_OF_A", "NONE"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    bridging_concepts: List[str] = Field(default_factory=list)


# =============================================================================
# 3. ConceptDistillerAgent (Bare-Minimum vs Self-Discovery Topics)
# =============================================================================
class ConceptDistillerInput(BaseModel):
    problem_statements: List[str]
    chunk_excerpts: List[str]


class ConceptTopicItem(BaseModel):
    title: str
    summary_md: str


class DistilledConceptsOutput(BaseModel):
    bare_minimum_topics: List[ConceptTopicItem]
    self_discovery_topics: List[ConceptTopicItem]
    anti_spoiler_verified: bool


# =============================================================================
# 4. HandwritingOCRAgent (Handwritten Image -> LaTeX/Markdown Preview)
# =============================================================================
class HandwritingOCRInput(BaseModel):
    problem_statement: str
    image_path: str


class HandwritingOCROutput(BaseModel):
    transcribed_latex_md: str
    detected_equations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_notes: Optional[str] = None


# =============================================================================
# 5. SolutionAuditorAgent (Self-Discovery Verification & Remedial Coaching)
# =============================================================================
class SolutionAuditInput(BaseModel):
    problem_statements: List[str]
    target_self_discovery_topics: List[str]
    learner_solutions_md: List[str]
    reference_hints_or_solutions: List[str] = Field(default_factory=list)


class TopicCoachingItem(BaseModel):
    topic: str
    explanation_md: str


class SolutionAuditOutput(BaseModel):
    outcome: Literal["SOLVED_WITH_DISCOVERY", "SOLVED_MISSED_DISCOVERY", "FAILED"]
    score_fraction: float = Field(ge=0.0, le=1.0)
    discovered_topics: List[str] = Field(default_factory=list)
    undiscovered_topics_to_teach: List[TopicCoachingItem] = Field(default_factory=list)
    remedial_scaffolding_topics: List[TopicCoachingItem] = Field(default_factory=list)
    feedback_summary_md: str


# =============================================================================
# Base Micro-Agent Definition
# =============================================================================
class MicroAgent(Generic[InputT, OutputT]):
    name: str = "BaseMicroAgent"
    schema_version: str = "1.0"
    role: ModelRole = "instruct"
    output_model: Type[OutputT]
    max_tokens: int = 512

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def build_user_prompt(self, inp: InputT) -> str:
        raise NotImplementedError

    def get_image_path(self, inp: InputT) -> Optional[Path]:
        return None

    def validate_semantics(self, inp: InputT, out: OutputT) -> Optional[str]:
        """Return an error string if semantic constraints are violated, else None."""
        return None


class ProblemBoundaryAgent(MicroAgent[ProblemBoundaryInput, ProblemBoundaryOutput]):
    name = "ProblemBoundaryAgent"
    schema_version = "1.0"
    role = "instruct"
    output_model = ProblemBoundaryOutput
    max_tokens = 768

    def build_system_prompt(self) -> str:
        return (
            "You are GRAPHY Companion's Problem Boundary Agent for Math, Physics, and CS problem sets. "
            "Extract individual problems into clean Markdown + LaTeX. "
            "CRITICAL RULE: Keep multi-part problems (e.g., (a), (b), (c)) together inside a SINGLE problem unit "
            "along with their shared stem context. Quarantine any inline hints or solutions into separate fields."
        )

    def build_user_prompt(self, inp: ProblemBoundaryInput) -> str:
        return f"Source: {inp.source_name}\n\nRaw Problem Text:\n{inp.raw_block_text}"

    def validate_semantics(self, inp: ProblemBoundaryInput, out: ProblemBoundaryOutput) -> Optional[str]:
        for p in out.problems:
            if not p.statement_md.strip():
                return f"Problem '{p.original_number}' has an empty statement_md."
        return None


class PrerequisiteJudgeAgent(MicroAgent[PrerequisitePairInput, PrerequisiteDecision]):
    name = "PrerequisiteJudgeAgent"
    schema_version = "1.0"
    role = "instruct"
    output_model = PrerequisiteDecision
    max_tokens = 256

    def build_system_prompt(self) -> str:
        return (
            "You are GRAPHY's Prerequisite Judge Agent for technical textbooks (Math, Physics, CS). "
            "Given Chunk A and Chunk B, determine whether understanding Chunk A is a strict prerequisite "
            "for understanding Chunk B ('A_PREREQ_OF_B'), whether Chunk B is a prerequisite for Chunk A "
            "('B_PREREQ_OF_A'), or neither ('NONE'). Output valid JSON."
        )

    def build_user_prompt(self, inp: PrerequisitePairInput) -> str:
        return (
            f"=== CHUNK A ({inp.chunk_a_id} | {inp.chunk_a_breadcrumb}) ===\n{inp.chunk_a_text}\n\n"
            f"=== CHUNK B ({inp.chunk_b_id} | {inp.chunk_b_breadcrumb}) ===\n{inp.chunk_b_text}"
        )

    def validate_semantics(self, inp: PrerequisitePairInput, out: PrerequisiteDecision) -> Optional[str]:
        if inp.chunk_a_id == inp.chunk_b_id and out.direction != "NONE":
            return "Identical chunks cannot be prerequisites of themselves; direction must be NONE."
        if not out.rationale.strip():
            return "Rationale must not be empty."
        return None


class ConceptDistillerAgent(MicroAgent[ConceptDistillerInput, DistilledConceptsOutput]):
    name = "ConceptDistillerAgent"
    schema_version = "1.0"
    role = "instruct"
    output_model = DistilledConceptsOutput
    max_tokens = 768

    def build_system_prompt(self) -> str:
        return (
            "You are GRAPHY's Pedagogical Concept Distiller Agent. "
            "Given a batch of problems and their textbook chunks, split the concepts into:\n"
            "1. `bare_minimum_topics`: Only the baseline definitions, notation, and foundational axioms "
            "needed to start tackling the problems.\n"
            "2. `self_discovery_topics`: The deeper theorems, properties, or derivation insights that the "
            "learner must self-discover by solving the problems.\n"
            "NEVER leak direct problem solutions in `bare_minimum_topics`."
        )

    def build_user_prompt(self, inp: ConceptDistillerInput) -> str:
        probs = "\n---\n".join(inp.problem_statements)
        chunks = "\n---\n".join(inp.chunk_excerpts)
        return f"=== TARGET PROBLEMS ===\n{probs}\n\n=== TEXTBOOK CHUNKS ===\n{chunks}"

    def validate_semantics(self, inp: ConceptDistillerInput, out: DistilledConceptsOutput) -> Optional[str]:
        if not out.bare_minimum_topics:
            return "bare_minimum_topics cannot be empty."
        if not out.self_discovery_topics:
            return "self_discovery_topics cannot be empty."
        bare_titles = {t.title.strip().lower() for t in out.bare_minimum_topics}
        disc_titles = {t.title.strip().lower() for t in out.self_discovery_topics}
        overlap = bare_titles.intersection(disc_titles)
        if overlap:
            return f"bare_minimum_topics and self_discovery_topics must be disjoint; overlap={overlap}"
        return None


class HandwritingOCRAgent(MicroAgent[HandwritingOCRInput, HandwritingOCROutput]):
    name = "HandwritingOCRAgent"
    schema_version = "1.0"
    role = "vision"
    output_model = HandwritingOCROutput
    max_tokens = 640

    def build_system_prompt(self) -> str:
        return (
            "You are GRAPHY's Handwritten Math/Physics/CS OCR Agent. "
            "Transcribe the user's handwritten work from the image into accurate Markdown and LaTeX "
            "(using $...$ for inline math and $$...$$ for display equations). Preserve every derivation step."
        )

    def build_user_prompt(self, inp: HandwritingOCRInput) -> str:
        return f"Problem context:\n{inp.problem_statement}\n\nTranscribe the handwritten solution image into LaTeX/Markdown."

    def get_image_path(self, inp: HandwritingOCRInput) -> Optional[Path]:
        return Path(inp.image_path)

    def validate_semantics(self, inp: HandwritingOCRInput, out: HandwritingOCROutput) -> Optional[str]:
        if not out.transcribed_latex_md.strip():
            return "transcribed_latex_md must not be empty."
        return None


class SolutionAuditorAgent(MicroAgent[SolutionAuditInput, SolutionAuditOutput]):
    name = "SolutionAuditorAgent"
    schema_version = "1.0"
    role = "instruct"
    output_model = SolutionAuditOutput
    max_tokens = 896

    def build_system_prompt(self) -> str:
        return (
            "You are GRAPHY's Solution Auditor & Remedial Coach Agent.\n"
            "Analyze the learner's solutions against the problems and target self-discovery topics:\n"
            "- If the learner solved the problems AND self-discovered all target topics: outcome='SOLVED_WITH_DISCOVERY'.\n"
            "- If the learner solved the problems but missed self-discovering some target topics: "
            "outcome='SOLVED_MISSED_DISCOVERY' and populate `undiscovered_topics_to_teach`.\n"
            "- If the learner failed to solve the problems: outcome='FAILED', populate BOTH "
            "`undiscovered_topics_to_teach` AND `remedial_scaffolding_topics` so the learner will not struggle."
        )

    def build_user_prompt(self, inp: SolutionAuditInput) -> str:
        return (
            f"=== PROBLEMS ===\n" + "\n---\n".join(inp.problem_statements) + "\n\n"
            f"=== TARGET SELF-DISCOVERY TOPICS ===\n" + "\n".join(f"- {t}" for t in inp.target_self_discovery_topics) + "\n\n"
            f"=== LEARNER SOLUTIONS ===\n" + "\n---\n".join(inp.learner_solutions_md)
        )

    def validate_semantics(self, inp: SolutionAuditInput, out: SolutionAuditOutput) -> Optional[str]:
        if out.outcome == "SOLVED_MISSED_DISCOVERY" and not out.undiscovered_topics_to_teach:
            return "When outcome is SOLVED_MISSED_DISCOVERY, undiscovered_topics_to_teach must be non-empty."
        if out.outcome == "FAILED" and (
            not out.undiscovered_topics_to_teach or not out.remedial_scaffolding_topics
        ):
            return "When outcome is FAILED, both undiscovered_topics_to_teach and remedial_scaffolding_topics must be provided."
        return None

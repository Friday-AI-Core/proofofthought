"""ProofOfThought: Main API for Z3-based reasoning with robust postprocessing.

This module provides a high-level interface for generating Z3 programs from
natural language questions, executing them via a selected backend, and
postprocessing the results to improve reliability and correctness.

Key improvements:
- Stronger error handling and structured logging
- Tight integration of self-consistency postprocessor in the answer flow
- Configurable postprocessing strategies (sequence, until-success, majority-vote)
- Richer QueryResult with error traces and postprocessing summary

Public interfaces remain backward compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence, Callable
import json
import logging
import os
import tempfile
import traceback

from z3adapter.reasoning.program_generator import Z3ProgramGenerator

if TYPE_CHECKING:
    from z3adapter.backends.abstract import Backend
    from z3adapter.postprocessors.abstract import Postprocessor

logger = logging.getLogger(__name__)

BackendType = Literal["json", "smt2"]


@dataclass
class QueryResult:
    """Result of a reasoning query.

    The original fields are preserved for backward compatibility.
    Added fields provide richer diagnostics and postprocessing insights.
    """

    # Existing, public fields (do not remove or rename)
    question: str
    answer: bool | None
    json_program: dict[str, Any] | None
    sat_count: int
    unsat_count: int
    output: str
    success: bool
    num_attempts: int

    # Pre-existing optional field that existed previously in some variants
    error: str | None = None

    # New fields for richer diagnostics and postprocessing introspection
    error_traces: list[str] = field(default_factory=list)
    postprocessing_summary: dict[str, Any] = field(default_factory=dict)
    # Confidence score in [0,1] where applicable (e.g., self-consistency)
    confidence: float | None = None
    # Raw backend artifacts to facilitate debugging (paths, snippets, etc.)
    artifacts: dict[str, Any] = field(default_factory=dict)
    # Trace log messages at important steps
    trace_log: list[str] = field(default_factory=list)

    def add_error(self, msg: str, exc: BaseException | None = None) -> None:
        self.error = msg if self.error is None else self.error
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.error_traces.append(tb)
            logger.debug("Captured exception trace", extra={"error": msg})
        else:
            self.error_traces.append(msg)

    def log(self, message: str, level: int = logging.INFO) -> None:
        self.trace_log.append(message)
        logger.log(level, message)


class ProofOfThought:
    """High-level API for Z3-based reasoning.

    Responsibilities:
    - Generate a JSON-DSL or SMT2 program from a question (via Z3ProgramGenerator)
    - Execute it via the configured backend
    - Integrate postprocessors to improve answers (self-consistency, etc.)

    Parameters
    ----------
    backend: Backend
        Concrete backend capable of running the generated program.
    generator: Z3ProgramGenerator | None
        Program generator. If not provided, a default instance is created.
    postprocessors: Sequence[Postprocessor] | None
        Postprocessors to apply in the given order.
    postprocessing_strategy: str | Callable
        Strategy for applying postprocessors. Supported built-ins:
        - "sequential": apply all in order, always keep latest success (default)
        - "until_success": stop at first postprocessor that yields success=True
        - "majority_vote": use postprocessors to produce multiple votes and
          select the majority answer (ties fall back to latest success)
        Can also be a callable with signature (question, initial_result, self,
        **kwargs) -> QueryResult
    cache_dir: str | None
        Folder to persist intermediate artifacts.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        generator: Z3ProgramGenerator | None = None,
        postprocessors: Sequence[Postprocessor] | None = None,
        postprocessing_strategy: str | Callable[..., QueryResult] = "sequential",
        cache_dir: str | None = None,
    ) -> None:
        self.backend = backend
        self.generator = generator or Z3ProgramGenerator()
        self.postprocessors: list[Postprocessor] = list(postprocessors or [])
        self.postprocessing_strategy = postprocessing_strategy
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="pot_cache_")

        # Ensure self-consistency postprocessor is tightly integrated when present
        # by moving it to the last position unless strategy dictates otherwise.
        try:
            sc_idx = next(
                (i for i, p in enumerate(self.postprocessors) if getattr(p, "name", "").lower() == "self_consistency"),
                None,
            )
            if sc_idx is not None and isinstance(self.postprocessing_strategy, str) and self.postprocessing_strategy != "majority_vote":
                self.postprocessors.append(self.postprocessors.pop(sc_idx))
        except Exception as e:
            logger.warning("Failed to reorder self-consistency postprocessor: %s", e)

    # ---------------------------- Core API ---------------------------- #

    def query(
        self,
        question: str,
        *,
        backend_type: BackendType = "json",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        attempts: int = 1,
    ) -> QueryResult:
        """Run reasoning for a question and return a QueryResult.

        Public signature is preserved. Added robustness and richer results.
        """
        result = QueryResult(
            question=question,
            answer=None,
            json_program=None,
            sat_count=0,
            unsat_count=0,
            output="",
            success=False,
            num_attempts=0,
        )

        result.log(f"Starting query. backend_type={backend_type}, attempts={attempts}")

        try:
            program_json, artifacts = self._generate_program(question, backend_type)
            result.json_program = program_json
            result.artifacts.update(artifacts)
            result.log("Program generation completed.")
        except Exception as e:
            msg = f"Program generation failed: {e}"
            logger.exception(msg)
            result.add_error(msg, e)
            return self._finalize_with_postprocessing(
                question=question,
                initial_result=result,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Execute with basic retry loop
        for i in range(1, max(1, attempts) + 1):
            result.num_attempts = i
            try:
                exec_out = self._execute_program(result.json_program, backend_type)
                result.output = exec_out.get("raw", result.output)
                result.sat_count += exec_out.get("sat_count", 0)
                result.unsat_count += exec_out.get("unsat_count", 0)
                result.answer = exec_out.get("answer", result.answer)
                result.success = exec_out.get("success", False)
                result.log(f"Attempt {i} execution complete. success={result.success}")
                if result.success:
                    break
            except Exception as e:
                msg = f"Execution failed on attempt {i}: {e}"
                logger.exception(msg)
                result.add_error(msg, e)

        # Apply postprocessing according to configured strategy
        result = self._apply_postprocessing_strategy(
            strategy=self.postprocessing_strategy,
            question=question,
            initial_result=result,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return result

    # ------------------------- Internal Helpers ------------------------- #

    def _generate_program(self, question: str, backend_type: BackendType) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a program representation using the generator.

        Returns the program and a dict of artifacts/paths for debugging.
        """
        logger.info("Generating program for question")
        if backend_type == "json":
            program = self.generator.generate_json(question)
        elif backend_type == "smt2":
            program = self.generator.generate_smt2(question)
        else:
            raise ValueError(f"Unsupported backend_type: {backend_type}")

        artifacts: dict[str, Any] = {}
        try:
            # Persist a copy for offline debugging
            fd, path = tempfile.mkstemp(prefix="pot_prog_", suffix=f".{backend_type}", dir=self.cache_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if backend_type == "json":
                    json.dump(program, f, indent=2)
                else:
                    f.write(program if isinstance(program, str) else str(program))
            artifacts["program_path"] = path
        except Exception as e:
            logger.warning("Failed to persist generated program: %s", e)

        return program, artifacts

    def _execute_program(self, program: dict[str, Any] | str | None, backend_type: BackendType) -> dict[str, Any]:
        """Execute the generated program using the backend and summarize results."""
        if program is None:
            raise ValueError("No program available for execution")

        logger.info("Executing program with backend")
        return self.backend.run(program, format=backend_type)

    # ---------------------- Postprocessing Orchestration ---------------------- #

    def _apply_postprocessing_strategy(
        self,
        *,
        strategy: str | Callable[..., QueryResult],
        question: str,
        initial_result: QueryResult,
        temperature: float,
        max_tokens: int,
    ) -> QueryResult:
        if callable(strategy):
            return strategy(
                question=question,
                initial_result=initial_result,
                self=self,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if strategy == "until_success":
            return self._postprocess_until_success(question, initial_result, temperature, max_tokens)
        if strategy == "majority_vote":
            return self._postprocess_majority_vote(question, initial_result, temperature, max_tokens)
        # default: sequential
        return self._postprocess_sequential(question, initial_result, temperature, max_tokens)

    def _postprocess_sequential(
        self,
        question: str,
        initial_result: QueryResult,
        temperature: float,
        max_tokens: int,
    ) -> QueryResult:
        """Apply all configured postprocessors in order, keeping latest success."""
        current = initial_result
        details: list[dict[str, Any]] = []
        for postprocessor in self.postprocessors:
            name = getattr(postprocessor, "name", postprocessor.__class__.__name__)
            logger.info("Applying postprocessor: %s", name)
            try:
                enhanced = postprocessor.process(
                    question=question,
                    initial_result=current,
                    generator=self.generator,
                    backend=self.backend,
                    llm_client=getattr(self, "llm_client", None),
                    cache_dir=self.cache_dir,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                details.append({
                    "name": name,
                    "success": getattr(enhanced, "success", False),
                    "answer": getattr(enhanced, "answer", None),
                    "confidence": getattr(enhanced, "confidence", None),
                })
                if enhanced.success:
                    logger.info("Postprocessor %s produced a successful result", name)
                    current = enhanced
                else:
                    logger.warning("Postprocessor %s did not improve result; keeping current", name)
            except Exception as e:
                msg = f"Error in postprocessor {name}: {e}"
                logger.exception(msg)
                current.add_error(msg, e)
        current.postprocessing_summary["strategy"] = "sequential"
        current.postprocessing_summary["steps"] = details
        return current

    def _postprocess_until_success(
        self,
        question: str,
        initial_result: QueryResult,
        temperature: float,
        max_tokens: int,
    ) -> QueryResult:
        """Apply postprocessors until one succeeds; return immediately on success."""
        current = initial_result
        details: list[dict[str, Any]] = []
        for postprocessor in self.postprocessors:
            name = getattr(postprocessor, "name", postprocessor.__class__.__name__)
            logger.info("Applying postprocessor: %s", name)
            try:
                enhanced = postprocessor.process(
                    question=question,
                    initial_result=current,
                    generator=self.generator,
                    backend=self.backend,
                    llm_client=getattr(self, "llm_client", None),
                    cache_dir=self.cache_dir,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                details.append({
                    "name": name,
                    "success": getattr(enhanced, "success", False),
                    "answer": getattr(enhanced, "answer", None),
                    "confidence": getattr(enhanced, "confidence", None),
                })
                if enhanced.success:
                    enhanced.postprocessing_summary["strategy"] = "until_success"
                    enhanced.postprocessing_summary["steps"] = details
                    return enhanced
                else:
                    logger.info("No success; continuing to next postprocessor")
            except Exception as e:
                msg = f"Error in postprocessor {name}: {e}"
                logger.exception(msg)
                current.add_error(msg, e)
        current.postprocessing_summary["strategy"] = "until_success"
        current.postprocessing_summary["steps"] = details
        return current

    def _postprocess_majority_vote(
        self,
        question: str,
        initial_result: QueryResult,
        temperature: float,
        max_tokens: int,
    ) -> QueryResult:
        """Use postprocessors to obtain multiple votes and pick the majority.

        Self-consistency postprocessors are expected to provide multiple samples
        internally and set `confidence` to the fraction agreeing with the final
        answer. If not provided, this method will compute a simple majority over
        available postprocessor outputs.
        """
        votes: list[tuple[bool | None, float | None, QueryResult, str]] = []
        details: list[dict[str, Any]] = []
        current = initial_result
        for postprocessor in self.postprocessors:
            name = getattr(postprocessor, "name", postprocessor.__class__.__name__)
            try:
                enhanced = postprocessor.process(
                    question=question,
                    initial_result=current,
                    generator=self.generator,
                    backend=self.backend,
                    llm_client=getattr(self, "llm_client", None),
                    cache_dir=self.cache_dir,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                conf = getattr(enhanced, "confidence", None)
                votes.append((enhanced.answer, conf, enhanced, name))
                details.append({"name": name, "answer": enhanced.answer, "confidence": conf, "success": enhanced.success})
            except Exception as e:
                msg = f"Error in postprocessor {name}: {e}"
                logger.exception(msg)
                current.add_error(msg, e)
                details.append({"name": name, "error": str(e), "success": False})

        # Tally votes for True/False; ignore None in majority count
        tally = {True: 0, False: 0}
        weight = {True: 0.0, False: 0.0}
        for ans, conf, _, _ in votes:
            if ans is None:
                continue
            tally[ans] += 1
            weight[ans] += conf if conf is not None else 1.0

        # Decide by weighted majority if weights differ, else by raw count
        decided: bool | None
        if weight[True] > weight[False]:
            decided = True
        elif weight[False] > weight[True]:
            decided = False
        elif tally[True] > tally[False]:
            decided = True
        elif tally[False] > tally[True]:
            decided = False
        else:
            decided = current.answer  # tie-breaker: keep current

        # Pick the best enhanced result corresponding to the decided answer
        best: QueryResult | None = None
        best_conf = -1.0
        for ans, conf, enhanced, name in votes:
            if ans == decided:
                c = conf if conf is not None else 0.0
                if c > best_conf:
                    best_conf = c
                   

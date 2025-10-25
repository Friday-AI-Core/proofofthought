"""Self-Consistency postprocessor for improving answer reliability.
Based on the Self-Consistency technique from:
"Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
"""
import json
import logging
import os
import tempfile
from collections import Counter
from typing import TYPE_CHECKING, Any

from z3adapter.postprocessors.abstract import Postprocessor

if TYPE_CHECKING:
    from z3adapter.backends.abstract import Backend
    from z3adapter.reasoning.program_generator import Z3ProgramGenerator
    from z3adapter.reasoning.proof_of_thought import QueryResult

logger = logging.getLogger(__name__)


class SelfConsistency(Postprocessor):
    """Self-Consistency postprocessor using majority voting.
    
    The Self-Consistency technique works by:
    1. Generating multiple independent reasoning paths
    2. Collecting answers from all paths
    3. Selecting the most consistent answer via majority voting
    
    This increases reliability by reducing the impact of random errors
    or spurious reasoning in any single attempt.
    """

    def __init__(self, num_samples: int = 5, name: str | None = None):
        """Initialize Self-Consistency postprocessor.
        
        Args:
            num_samples: Number of independent samples to generate
            name: Optional custom name for this postprocessor
        """
        super().__init__(name)
        self.num_samples = num_samples

    def process(
        self,
        question: str,
        initial_result: "QueryResult",
        generator: "Z3ProgramGenerator",
        backend: "Backend",
        llm_client: Any,
        **kwargs: Any,
    ) -> "QueryResult":
        """Apply Self-Consistency to improve answer reliability.
        
        Args:
            question: Original question
            initial_result: Initial QueryResult
            generator: Program generator
            backend: Execution backend
            llm_client: LLM client for generation
            **kwargs: Additional arguments
            
        Returns:
            QueryResult with the most consistent answer
        """
        logger.info(f"[{self.name}] Starting self-consistency with {self.num_samples} samples")
        
        # Collect all samples including the initial result
        all_results = [initial_result]
        
        # Generate additional samples
        for i in range(self.num_samples - 1):
            logger.info(f"[{self.name}] Generating sample {i + 2}/{self.num_samples}")
            try:
                sample_result = self._generate_sample(
                    question, generator, backend, llm_client, **kwargs
                )
                all_results.append(sample_result)
            except Exception as e:
                logger.error(f"[{self.name}] Failed to generate sample {i + 2}: {e}")
        
        # Filter successful results with valid answers
        valid_results = [
            r for r in all_results 
            if r.success and r.answer is not None
        ]
        
        if not valid_results:
            logger.warning(f"[{self.name}] No valid results found, returning initial result")
            return initial_result
        
        # Aggregate answers and select the most consistent one
        answer_counts = Counter(r.answer for r in valid_results)
        most_common_answers = answer_counts.most_common()
        
        logger.info(f"[{self.name}] Answer distribution: {dict(answer_counts)}")
        
        # Handle tie-breaking with confidence-based selection
        if len(most_common_answers) > 1 and most_common_answers[0][1] == most_common_answers[1][1]:
            # Tie detected - select based on result quality metrics
            logger.info(f"[{self.name}] Tie detected, using confidence-based tie-breaking")
            tied_answer = self._break_tie(valid_results, most_common_answers)
        else:
            tied_answer = most_common_answers[0][0]
        
        # Find the best result with the most consistent answer
        best_result = self._select_best_result(
            [r for r in valid_results if r.answer == tied_answer]
        )
        
        logger.info(
            f"[{self.name}] Selected answer '{best_result.answer}' "
            f"(appeared {answer_counts[best_result.answer]}/{len(valid_results)} times)"
        )
        
        return best_result

    def _generate_sample(
        self,
        question: str,
        generator: "Z3ProgramGenerator",
        backend: "Backend",
        llm_client: Any,
        **kwargs: Any,
    ) -> "QueryResult":
        """Generate a single sample result.
        
        Args:
            question: Question to answer
            generator: Program generator
            backend: Execution backend
            llm_client: LLM client
            **kwargs: Additional arguments
            
        Returns:
            QueryResult for this sample
        """
        try:
            # Generate Z3 program
            gen_result = generator.generate(question, llm_client)
            
            if not gen_result.success or not gen_result.json_program:
                return QueryResult(
                    question=question,
                    answer=None,
                    json_program=None,
                    sat_count=0,
                    unsat_count=0,
                    output="",
                    success=False,
                    num_attempts=0,
                )
            
            # Execute the program
            verify_result = backend.verify(gen_result.json_program)
            
            return QueryResult(
                question=question,
                answer=verify_result.answer,
                json_program=gen_result.json_program,
                sat_count=verify_result.sat_count,
                unsat_count=verify_result.unsat_count,
                output=verify_result.output,
                success=verify_result.success and verify_result.answer is not None,
                num_attempts=1,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Error generating sample: {e}")
            return QueryResult(
                question=question,
                answer=None,
                json_program=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                num_attempts=0,
                error=str(e),
            )

    def _break_tie(
        self,
        valid_results: list["QueryResult"],
        tied_answers: list[tuple[Any, int]],
    ) -> Any:
        """Break ties between equally frequent answers using confidence metrics.
        
        Args:
            valid_results: All valid results
            tied_answers: List of (answer, count) tuples with equal counts
            
        Returns:
            The answer selected after tie-breaking
        """
        # Get all answers that are tied for most common
        max_count = tied_answers[0][1]
        tied_answer_values = [ans for ans, count in tied_answers if count == max_count]
        
        # Calculate average confidence metrics for each tied answer
        answer_metrics = {}
        for answer in tied_answer_values:
            matching_results = [r for r in valid_results if r.answer == answer]
            
            # Use SAT/UNSAT clarity as a confidence metric
            # Higher clarity = more confident answer
            avg_clarity = sum(
                abs(r.sat_count - r.unsat_count) for r in matching_results
            ) / len(matching_results)
            
            # Prefer results with fewer attempts (cleaner generation)
            avg_attempts = sum(r.num_attempts for r in matching_results) / len(matching_results)
            
            # Combined score: higher clarity is better, fewer attempts is better
            answer_metrics[answer] = (avg_clarity, -avg_attempts)
        
        # Select answer with best metrics
        best_answer = max(answer_metrics, key=answer_metrics.get)
        logger.info(f"[{self.name}] Tie-breaking metrics: {answer_metrics}")
        
        return best_answer

    def _select_best_result(
        self,
        matching_results: list["QueryResult"],
    ) -> "QueryResult":
        """Select the best result from a list of results with the same answer.
        
        Args:
            matching_results: Results with the same answer
            
        Returns:
            The best QueryResult based on quality metrics
        """
        if not matching_results:
            raise ValueError("No matching results to select from")
        
        # Sort by quality: prefer cleaner generation (fewer attempts) and clearer SAT/UNSAT
        best_result = max(
            matching_results,
            key=lambda r: (
                -r.num_attempts,  # Fewer attempts is better (negative for max)
                abs(r.sat_count - r.unsat_count),  # Higher clarity is better
            )
        )
        
        return best_result

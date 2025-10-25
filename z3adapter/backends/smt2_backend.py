"""SMT2 backend using Z3 command-line interface."""
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from z3adapter.backends.abstract import Backend, VerificationResult
from z3adapter.reasoning.smt2_prompt_template import SMT2_INSTRUCTIONS

logger = logging.getLogger(__name__)


class SMT2Backend(Backend):
    """Backend for executing SMT2 programs via Z3 CLI with enhanced reliability."""

    def __init__(
        self,
        verify_timeout: int = 10000,
        z3_path: str = "z3",
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ) -> None:
        """Initialize SMT2 backend with configurable timeout and retry logic.

        Args:
            verify_timeout: Timeout for verification in milliseconds (default: 10000)
            z3_path: Path to Z3 executable (default: "z3" from PATH)
            max_retries: Maximum number of retry attempts on failure (default: 2)
            retry_delay: Delay in seconds between retry attempts (default: 0.5)

        Raises:
            FileNotFoundError: If Z3 executable is not found
        """
        self.verify_timeout = verify_timeout
        self.z3_path = z3_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Validate Z3 is available
        if not shutil.which(z3_path):
            raise FileNotFoundError(
                f"Z3 executable not found: '{z3_path}'\n"
                f"Please install Z3:\n"
                f"  - pip install z3-solver\n"
                f"  - Or download from: https://github.com/Z3Prover/z3/releases\n"
                f"  - Or specify custom path: SMT2Backend(z3_path='/path/to/z3')"
            )

        logger.info(
            f"Initialized SMT2Backend: z3_path={z3_path}, "
            f"timeout={verify_timeout}ms, max_retries={max_retries}"
        )

    def execute(self, program_path: str) -> VerificationResult:
        """Execute an SMT2 program via Z3 CLI with robust error handling.

        This method invokes Z3 on the given SMT2 program with comprehensive:
        - Process stdout/stderr capture
        - Exit code checking
        - Timeout handling
        - Retry logic on transient failures
        - Detailed diagnostic logging

        Args:
            program_path: Path to SMT2 program file

        Returns:
            VerificationResult with answer, counts, output, and diagnostic information
        """
        program_path_obj = Path(program_path)
        if not program_path_obj.exists():
            error_msg = f"Program file not found: {program_path}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

        logger.info(f"Executing SMT2 program: {program_path}")

        # Attempt execution with retry logic
        last_result = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.warning(
                    f"Retry attempt {attempt}/{self.max_retries} after failure"
                )
                time.sleep(self.retry_delay)

            result = self._execute_once(program_path)

            # If successful or hard error (non-transient), return immediately
            if result.success or self._is_hard_error(result):
                return result

            last_result = result

        # All retries exhausted
        logger.error(
            f"All {self.max_retries + 1} execution attempts failed for {program_path}"
        )
        return last_result

    def _execute_once(self, program_path: str) -> VerificationResult:
        """Execute Z3 once on the given program with full diagnostics.

        Args:
            program_path: Path to SMT2 program file

        Returns:
            VerificationResult with comprehensive execution details
        """
        # Build Z3 command with timeout
        timeout_seconds = self.verify_timeout / 1000.0
        cmd = [self.z3_path, "-T:%d" % int(timeout_seconds), program_path]

        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            # Execute Z3 with captured stdout, stderr, and enforced timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 5,  # Add buffer to subprocess timeout
                check=False,  # Don't raise on non-zero exit; handle manually
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            exit_code = result.returncode

            # Log execution outcome
            logger.info(
                f"Z3 execution completed: exit_code={exit_code}, "
                f"stdout_len={len(stdout)}, stderr_len={len(stderr)}"
            )
            if stdout:
                logger.debug(f"Z3 stdout: {stdout[:500]}...")  # Log first 500 chars
            if stderr:
                logger.warning(f"Z3 stderr: {stderr}")

            # Check for errors
            if exit_code != 0:
                error_msg = (
                    f"Z3 exited with non-zero code {exit_code}. "
                    f"Stderr: {stderr if stderr else '(empty)'}"
                )
                logger.error(error_msg)
                return VerificationResult(
                    answer=None,
                    sat_count=0,
                    unsat_count=0,
                    output=stdout,
                    success=False,
                    error=error_msg,
                )

            # Parse output for sat/unsat counts
            sat_count, unsat_count = self._parse_z3_output(stdout)

            # Determine answer based on parsed results
            answer = self._determine_answer(sat_count, unsat_count, stdout)

            logger.info(
                f"Verification result: answer={answer}, "
                f"sat={sat_count}, unsat={unsat_count}"
            )

            return VerificationResult(
                answer=answer,
                sat_count=sat_count,
                unsat_count=unsat_count,
                output=stdout,
                success=True,
                error=None,
            )

        except subprocess.TimeoutExpired:
            error_msg = (
                f"Z3 execution timed out after {timeout_seconds}s. "
                f"Consider increasing verify_timeout or simplifying the problem."
            )
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

        except FileNotFoundError:
            error_msg = f"Z3 executable not found at runtime: {self.z3_path}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

        except Exception as e:
            error_msg = f"Unexpected error during Z3 execution: {type(e).__name__}: {e}"
            logger.exception(error_msg)  # Log with full traceback
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

    def _parse_z3_output(self, output: str) -> tuple[int, int]:
        """Parse Z3 output to count sat/unsat results with improved accuracy.

        Uses refined regex patterns to avoid false positives (e.g., 'unsat' vs 'sat').

        Args:
            output: Raw Z3 output text

        Returns:
            Tuple of (sat_count, unsat_count)
        """
        # Pattern: match "sat" only when NOT preceded by "un" (negative lookbehind)
        # Word boundary ensures we match whole words
        sat_pattern = r"(?<!un)\bsat\b"
        unsat_pattern = r"\bunsat\b"

        sat_matches = re.findall(sat_pattern, output, re.IGNORECASE)
        unsat_matches = re.findall(unsat_pattern, output, re.IGNORECASE)

        sat_count = len(sat_matches)
        unsat_count = len(unsat_matches)

        logger.debug(f"Parsed Z3 output: sat={sat_count}, unsat={unsat_count}")
        return sat_count, unsat_count

    def _determine_answer(self, sat_count: int, unsat_count: int, output: str) -> Optional[str]:
        """Determine the verification answer based on parsed counts and output.

        Provides fallback logic for ambiguous or empty outputs.

        Args:
            sat_count: Number of 'sat' occurrences
            unsat_count: Number of 'unsat' occurrences
            output: Raw Z3 output for fallback analysis

        Returns:
            'sat', 'unsat', 'unknown', or None if indeterminate
        """
        # Clear unsat dominates
        if unsat_count > 0 and sat_count == 0:
            return "unsat"

        # Clear sat (with no unsat)
        if sat_count > 0 and unsat_count == 0:
            return "sat"

        # Both present: ambiguous, but unsat typically appears in error messages
        # Prefer sat if it's the final result line
        if sat_count > 0 and unsat_count > 0:
            lines = output.strip().split("\n")
            if lines and "sat" in lines[-1].lower() and "unsat" not in lines[-1].lower():
                logger.debug("Ambiguous output; last line suggests 'sat'")
                return "sat"
            elif lines and "unsat" in lines[-1].lower():
                logger.debug("Ambiguous output; last line suggests 'unsat'")
                return "unsat"

        # Fallback: check for 'unknown' response
        if "unknown" in output.lower():
            logger.debug("Z3 returned 'unknown'")
            return "unknown"

        # No clear result
        logger.warning("Could not determine answer from Z3 output")
        return None

    def _is_hard_error(self, result: VerificationResult) -> bool:
        """Determine if an error is non-transient (hard) and should not be retried.

        Args:
            result: The verification result to check

        Returns:
            True if error is non-transient, False if it might be transient
        """
        if result.success or result.error is None:
            return False

        error_lower = result.error.lower()

        # Hard errors: file not found, parse errors, invalid syntax
        hard_error_indicators = [
            "file not found",
            "not found at runtime",
            "parse error",
            "syntax error",
            "invalid",
        ]

        return any(indicator in error_lower for indicator in hard_error_indicators)

    def get_file_extension(self) -> str:
        """Get the file extension for SMT2 programs.

        Returns:
            ".smt2"
        """
        return ".smt2"

    def get_prompt_template(self) -> str:
        """Get the prompt template for SMT2 generation.

        Returns:
            SMT2 prompt template for LLM-based program generation
        """
        return SMT2_INSTRUCTIONS

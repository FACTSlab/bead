"""jsPsych deployment components.

This module provides functionality for generating jsPsych experiments with
constraint-aware trial randomization.
"""

from bead.deployment.jspsych.randomizer import generate_randomizer_function

__all__ = ["generate_randomizer_function"]

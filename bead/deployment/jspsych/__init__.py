"""jsPsych 8.x deployment components.

Generates jsPsych experiments with batch mode support and server-side list
distribution via JATOS batch sessions.
"""

from bead.deployment.jspsych.randomizer import generate_randomizer_function

__all__ = ["generate_randomizer_function"]

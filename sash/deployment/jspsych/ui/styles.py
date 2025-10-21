"""Material Design CSS generation for jsPsych experiments.

This module provides the MaterialDesignStylesheet class for generating
Material Design 3 CSS for SASH jsPsych experiments.
"""

from __future__ import annotations

from typing import Literal


class MaterialDesignStylesheet:
    """Generator for Material Design 3 CSS.

    Generates Material Design 3 compliant CSS for jsPsych experiments.

    Features
    --------
    - Color theming (light/dark/auto)
    - Typography (Roboto font family)
    - Elevation (shadows)
    - Ripple effects for buttons
    - Form controls (inputs, dropdowns, radio buttons)

    Parameters
    ----------
    theme : Literal["light", "dark", "auto"]
        Color theme for the stylesheet.
    primary_color : str
        Primary color (hex code).
    secondary_color : str
        Secondary color (hex code).

    Examples
    --------
    >>> stylesheet = MaterialDesignStylesheet()
    >>> css = stylesheet.generate_css(theme="light")
    >>> print(css[:100])
    """

    def __init__(self) -> None:
        """Initialize Material Design stylesheet generator."""
        pass

    def generate_css(
        self,
        theme: Literal["light", "dark", "auto"] = "light",
        primary_color: str = "#6200EE",
        secondary_color: str = "#03DAC6",
    ) -> str:
        """Generate complete Material Design CSS.

        Parameters
        ----------
        theme : Literal["light", "dark", "auto"]
            Color theme (light, dark, or auto).
        primary_color : str
            Primary color as hex code.
        secondary_color : str
            Secondary color as hex code.

        Returns
        -------
        str
            Complete CSS stylesheet as string.

        Examples
        --------
        >>> stylesheet = MaterialDesignStylesheet()
        >>> css = stylesheet.generate_css(theme="light", primary_color="#1976D2")
        >>> "--primary-color" in css
        True
        """
        # Determine theme colors
        if theme == "light":
            background = "#FFFFFF"
            surface = "#FFFFFF"
            on_surface = "#000000"
            on_primary = "#FFFFFF"
        elif theme == "dark":
            background = "#121212"
            surface = "#1E1E1E"
            on_surface = "#FFFFFF"
            on_primary = "#000000"
        else:  # auto
            # Use CSS media query for system preference
            background = "#FFFFFF"
            surface = "#FFFFFF"
            on_surface = "#000000"
            on_primary = "#FFFFFF"

        css = f"""
/* Material Design 3 Stylesheet for SASH jsPsych Experiments */
/* Theme: {theme} */

@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

:root {{
  /* Color palette */
  --primary-color: {primary_color};
  --secondary-color: {secondary_color};
  --background: {background};
  --surface: {surface};
  --on-surface: {on_surface};
  --on-primary: {on_primary};
  --error: #B00020;
  --on-error: #FFFFFF;

  /* Typography */
  --font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-size-body: 16px;
  --font-size-title: 24px;
  --font-size-label: 14px;

  /* Spacing */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* Elevation shadows */
  --elevation-1: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
  --elevation-2: 0 3px 6px rgba(0, 0, 0, 0.16), 0 3px 6px rgba(0, 0, 0, 0.23);
  --elevation-3: 0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23);

  /* Border radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 9999px;
}}

/* Dark theme media query for auto mode */
@media (prefers-color-scheme: dark) {{
  :root {{
    --background: #121212;
    --surface: #1E1E1E;
    --on-surface: #FFFFFF;
    --on-primary: #000000;
  }}
}}

/* Base styles */
body {{
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  color: var(--on-surface);
  background-color: var(--background);
  margin: 0;
  padding: 0;
  line-height: 1.5;
}}

/* Button styles */
.sash-button {{
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  font-weight: 500;
  padding: 10px 24px;
  border: none;
  border-radius: var(--radius-sm);
  background-color: var(--primary-color);
  color: var(--on-primary);
  cursor: pointer;
  box-shadow: var(--elevation-2);
  transition: box-shadow 0.2s, background-color 0.2s;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.sash-button:hover {{
  box-shadow: var(--elevation-3);
}}

.sash-button:active {{
  box-shadow: var(--elevation-1);
}}

.sash-button:disabled {{
  background-color: rgba(0, 0, 0, 0.12);
  color: rgba(0, 0, 0, 0.38);
  box-shadow: none;
  cursor: not-allowed;
}}

/* Rating scale styles */
.sash-rating-container {{
  max-width: 800px;
  margin: 0 auto;
  padding: var(--spacing-xl);
}}

.sash-rating-prompt {{
  font-size: var(--font-size-title);
  font-weight: 500;
  margin-bottom: var(--spacing-lg);
  text-align: center;
}}

.sash-rating-scale {{
  display: flex;
  justify-content: center;
  gap: var(--spacing-sm);
  margin: var(--spacing-lg) 0;
}}

.sash-rating-option {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-xs);
}}

.sash-rating-button {{
  width: 48px;
  height: 48px;
  border: 2px solid var(--primary-color);
  border-radius: var(--radius-full);
  background-color: transparent;
  color: var(--primary-color);
  font-size: var(--font-size-body);
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
}}

.sash-rating-button:hover {{
  background-color: rgba(98, 0, 238, 0.08);
}}

.sash-rating-button.selected {{
  background-color: var(--primary-color);
  color: var(--on-primary);
}}

.sash-rating-label {{
  font-size: var(--font-size-label);
  color: var(--on-surface);
  text-align: center;
  max-width: 80px;
}}

.sash-rating-button-container {{
  display: flex;
  justify-content: center;
  margin-top: var(--spacing-xl);
}}

/* Text field styles */
.sash-text-field {{
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  padding: 12px 16px;
  border: 1px solid rgba(0, 0, 0, 0.38);
  border-radius: var(--radius-sm);
  background-color: transparent;
  color: var(--on-surface);
  transition: border-color 0.2s;
}}

.sash-text-field:focus {{
  outline: none;
  border-color: var(--primary-color);
  border-width: 2px;
}}

/* Dropdown styles */
.sash-dropdown {{
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  padding: 12px 16px;
  border: 1px solid rgba(0, 0, 0, 0.38);
  border-radius: var(--radius-sm);
  background-color: var(--surface);
  color: var(--on-surface);
  cursor: pointer;
  transition: border-color 0.2s;
}}

.sash-dropdown:focus {{
  outline: none;
  border-color: var(--primary-color);
  border-width: 2px;
}}

/* Radio group styles */
.sash-radio-group {{
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}}

.sash-radio-option {{
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color 0.2s;
}}

.sash-radio-option:hover {{
  background-color: rgba(0, 0, 0, 0.04);
}}

/* Card styles */
.sash-card {{
  background-color: var(--surface);
  border-radius: var(--radius-md);
  padding: var(--spacing-md);
  box-shadow: var(--elevation-1);
  transition: box-shadow 0.2s;
}}

.sash-card:hover {{
  box-shadow: var(--elevation-2);
}}

/* Progress indicator styles */
.sash-progress {{
  width: 100%;
  height: 4px;
  background-color: rgba(98, 0, 238, 0.12);
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000;
}}

.sash-progress-bar {{
  height: 100%;
  background-color: var(--primary-color);
  transition: width 0.3s;
}}

/* Cloze task styles */
.sash-cloze-container {{
  max-width: 900px;
  margin: 0 auto;
  padding: var(--spacing-xl);
}}

.sash-cloze-text {{
  font-size: var(--font-size-body);
  line-height: 2;
  margin: var(--spacing-lg) 0;
}}

.sash-cloze-field {{
  min-width: 120px;
  margin: 0 var(--spacing-xs);
  display: inline-block;
}}

.sash-cloze-button-container {{
  display: flex;
  justify-content: center;
  margin-top: var(--spacing-xl);
}}

/* Forced choice styles */
.sash-forced-choice-container {{
  max-width: 1000px;
  margin: 0 auto;
  padding: var(--spacing-xl);
}}

.sash-forced-choice-prompt {{
  font-size: var(--font-size-title);
  font-weight: 500;
  margin-bottom: var(--spacing-lg);
  text-align: center;
}}

.sash-forced-choice-alternatives {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-lg);
  margin: var(--spacing-lg) 0;
}}

.sash-alternative {{
  padding: var(--spacing-lg);
  transition: transform 0.2s, box-shadow 0.2s;
}}

.sash-alternative:hover {{
  transform: translateY(-2px);
}}

.sash-alternative.selected {{
  border: 2px solid var(--primary-color);
  box-shadow: var(--elevation-3);
}}

.sash-alternative-label {{
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--primary-color);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: var(--spacing-sm);
}}

.sash-alternative-content {{
  font-size: var(--font-size-body);
  margin-bottom: var(--spacing-md);
  min-height: 60px;
}}

.sash-choice-button {{
  width: 100%;
}}
"""
        return css

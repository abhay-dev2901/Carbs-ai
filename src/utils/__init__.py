# src/utils/__init__.py
"""
Utility functions for the Indian Food Carb Estimator project.
"""

from .diabetic_check import (
    analyze_food_for_diabetics,
    display_diabetic_safety_ratings,
    SAMPLE_FOODS
)

__all__ = [
    'analyze_food_for_diabetics',
    'display_diabetic_safety_ratings',
    'SAMPLE_FOODS'
]


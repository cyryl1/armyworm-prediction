"""
Structured pest management recommendations.

This module keeps recommendation logic separate from model inference so the
lookup strategy can evolve without touching the detector pipeline.
"""

from __future__ import annotations

from typing import Dict


RECOMMENDATION_RULES: Dict[str, Dict[str, str]] = {
    "fall-armyworm-egg": {
        "severity": "low",
        "management_tier": "cultural",
        "primary_action": "Scout fields daily and remove egg masses where feasible.",
        "secondary_action": "Use threshold-based intervention if infestation expands.",
    },
    "fall-armyworm-frass": {
        "severity": "medium",
        "management_tier": "biological",
        "primary_action": "Inspect surrounding leaves for active larvae and damage.",
        "secondary_action": "Consider biological control before chemical treatment.",
    },
    "fall-armyworm-larva": {
        "severity": "high",
        "management_tier": "chemical",
        "primary_action": "Apply targeted control promptly using label-approved products.",
        "secondary_action": "Rotate modes of action to reduce resistance pressure.",
    },
    "fall-armyworm-larval-damage": {
        "severity": "medium",
        "management_tier": "integrated",
        "primary_action": "Assess live larvae before spraying and map affected areas.",
        "secondary_action": "Escalate treatment if fresh feeding is still active.",
    },
    "healthy-maize": {
        "severity": "none",
        "management_tier": "monitoring",
        "primary_action": "No treatment required.",
        "secondary_action": "Continue routine scouting.",
    },
    "maize-streak-disease": {
        "severity": "high",
        "management_tier": "integrated",
        "primary_action": "Rogue infected plants and control vector pressure.",
        "secondary_action": "Use resistant varieties for future plantings.",
    },
}


def get_recommendation_details(class_name: str) -> Dict[str, str]:
    """Return structured management guidance for a detected class."""
    return RECOMMENDATION_RULES.get(
        class_name,
        {
            "severity": "unknown",
            "management_tier": "monitoring",
            "primary_action": "Review the detection manually before acting.",
            "secondary_action": "Capture more context to refine the recommendation.",
        },
    )


def format_recommendation(class_name: str) -> str:
    """Return a compact human-readable recommendation string."""
    details = get_recommendation_details(class_name)
    return f"{details['management_tier'].title()} | {details['primary_action']}"

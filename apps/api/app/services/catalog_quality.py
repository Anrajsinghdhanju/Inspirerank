from __future__ import annotations

import re
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z]+")
VOWELS = set("aeiouy")


@dataclass(frozen=True)
class QualityAssessment:
    allowed: bool
    score: float
    reasons: tuple[str, ...]


def _safe(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def assess_item(metadata: dict) -> QualityAssessment:
    """
    Conservative catalog-quality heuristic.

    Goal: remove clearly broken/incomplete records without filtering legitimate
    long-tail products simply because they are uncommon.

    The ranking system can use `score` as a soft signal while `allowed` is only
    False for fairly obvious quality problems.
    """
    title = _safe(metadata.get("title"))
    search_text = _safe(metadata.get("search_text"))
    category = _safe(metadata.get("main_category"))
    image_url = _safe(metadata.get("image_url"))

    reasons: list[str] = []

    words = WORD_RE.findall(title)
    alpha_chars = sum(character.isalpha() for character in title)
    visible_chars = sum(not character.isspace() for character in title)
    alpha_ratio = alpha_chars / max(visible_chars, 1)

    suspicious_consonant_token = any(
        len(word) >= 7
        and not any(character.lower() in VOWELS for character in word)
        for word in words
    )

    if not title:
        reasons.append("missing_title")
    if len(title) < 5:
        reasons.append("title_too_short")
    if not words:
        reasons.append("no_alphabetic_words")
    if alpha_ratio < 0.45:
        reasons.append("low_alpha_ratio")
    if suspicious_consonant_token:
        reasons.append("suspicious_consonant_token")

    # Soft score. This is intentionally forgiving.
    score = 0.0

    if 10 <= len(title) <= 240:
        score += 0.30
    elif len(title) >= 5:
        score += 0.15

    if len(words) >= 3:
        score += 0.20
    elif len(words) >= 2:
        score += 0.12
    elif len(words) == 1 and len(title) >= 12:
        score += 0.05

    if len(search_text) >= 120:
        score += 0.25
    elif len(search_text) >= 40:
        score += 0.12

    if image_url:
        score += 0.15

    if category:
        score += 0.10

    if suspicious_consonant_token:
        score -= 0.35

    score = max(0.0, min(1.0, score))

    hard_fail_reasons = {
        "missing_title",
        "title_too_short",
        "no_alphabetic_words",
        "low_alpha_ratio",
        "suspicious_consonant_token",
    }

    allowed = not any(reason in hard_fail_reasons for reason in reasons)

    return QualityAssessment(
        allowed=allowed,
        score=score,
        reasons=tuple(reasons),
    )

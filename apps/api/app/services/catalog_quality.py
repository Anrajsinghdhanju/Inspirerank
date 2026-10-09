from __future__ import annotations

from dataclasses import dataclass


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


def _alpha_tokens(text: str) -> list[str]:
    """Unicode-aware alphabetic tokenization."""
    tokens: list[str] = []
    current: list[str] = []

    for character in text:
        if character.isalpha():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return tokens


def _is_suspicious_ascii_token(token: str) -> bool:
    return (
        token.isascii()
        and token.isalpha()
        and len(token) >= 7
        and not any(character.lower() in VOWELS for character in token)
    )


def _is_descriptive_token(token: str) -> bool:
    if not token:
        return False

    if not token.isascii():
        return True

    if len(token) < 3:
        return False

    return any(character.lower() in VOWELS for character in token)


def assess_item(metadata: dict) -> QualityAssessment:
    """Conservative, Unicode-aware catalog-quality heuristic."""
    title = _safe(metadata.get("title"))
    search_text = _safe(metadata.get("search_text"))
    category = _safe(metadata.get("main_category"))
    image_url = _safe(metadata.get("image_url"))

    reasons: list[str] = []

    tokens = _alpha_tokens(title)
    descriptive_tokens = [
        token for token in tokens if _is_descriptive_token(token)
    ]
    suspicious_tokens = [
        token for token in tokens if _is_suspicious_ascii_token(token)
    ]

    alpha_chars = sum(character.isalpha() for character in title)
    visible_chars = sum(not character.isspace() for character in title)
    alpha_ratio = alpha_chars / max(visible_chars, 1)

    if not title:
        reasons.append("missing_title")
    if len(title) < 5:
        reasons.append("title_too_short")
    if alpha_chars == 0:
        reasons.append("no_alphabetic_content")
    if alpha_ratio < 0.30 and len(descriptive_tokens) == 0:
        reasons.append("low_alpha_ratio")

    # Only hard-filter consonant-heavy corruption when there is very little
    # surrounding natural-language context. This retains legitimate brand/model
    # heavy product titles while still rejecting examples such as
    # "Tong 30 C dsrhgsdh".
    if (
        suspicious_tokens
        and len(descriptive_tokens) <= 1
        and len(tokens) <= 4
    ):
        reasons.append("suspicious_low_context_token")

    score = 0.0

    if 10 <= len(title) <= 240:
        score += 0.30
    elif len(title) >= 5:
        score += 0.15

    if len(descriptive_tokens) >= 3:
        score += 0.20
    elif len(descriptive_tokens) >= 1:
        score += 0.12

    if len(search_text) >= 120:
        score += 0.25
    elif len(search_text) >= 40:
        score += 0.12

    if image_url:
        score += 0.15
    if category:
        score += 0.10
    if "suspicious_low_context_token" in reasons:
        score -= 0.40

    score = max(0.0, min(1.0, score))

    hard_fail_reasons = {
        "missing_title",
        "title_too_short",
        "no_alphabetic_content",
        "low_alpha_ratio",
        "suspicious_low_context_token",
    }

    allowed = not any(reason in hard_fail_reasons for reason in reasons)

    return QualityAssessment(
        allowed=allowed,
        score=score,
        reasons=tuple(reasons),
    )

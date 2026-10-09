from app.services.catalog_quality import assess_item


def metadata(title: str) -> dict:
    return {
        "title": title,
        "search_text": (title + " arts crafts sewing ") * 8,
        "main_category": "Arts, Crafts & Sewing",
        "image_url": "https://example.com/x.jpg",
    }


def test_japanese_title_is_allowed():
    assessment = assess_item(
        metadata("ショウワグリム 折り紙 おりがみ ニューカラー千羽鶴用折紙 20-1274")
    )
    assert assessment.allowed


def test_model_code_heavy_title_is_allowed():
    assessment = assess_item(
        metadata("10 Piece Bobbin #0060265000 for Bernina 180 185 190 450")
    )
    assert assessment.allowed


def test_obvious_corruption_is_filtered():
    assessment = assess_item(metadata("Tong 30 C dsrhgsdh"))
    assert not assessment.allowed
    assert "suspicious_low_context_token" in assessment.reasons


def test_empty_title_is_filtered():
    assessment = assess_item(metadata(""))
    assert not assessment.allowed

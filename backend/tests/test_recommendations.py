"""Tests for recommendation logic."""

from app import recommendations


class TestRecommendations:
    """Test pest management recommendation generation."""

    def test_get_recommendation_details_armyworm_larva(self):
        """Should return high severity for active larvae."""
        details = recommendations.get_recommendation_details("fall-armyworm-larva")
        assert details["severity"] == "high"
        assert details["management_tier"] == "chemical"
        assert "Apply" in details["primary_action"]

    def test_get_recommendation_details_healthy(self):
        """Should return no action for healthy maize."""
        details = recommendations.get_recommendation_details("healthy-maize")
        assert details["severity"] == "none"
        assert details["management_tier"] == "monitoring"

    def test_get_recommendation_details_unknown_class(self):
        """Should return sensible defaults for unknown class."""
        details = recommendations.get_recommendation_details("unknown-pest")
        assert details["severity"] == "unknown"
        assert "Review" in details["primary_action"]

    def test_format_recommendation_compact(self):
        """Should format as compact one-liner."""
        rec = recommendations.format_recommendation("fall-armyworm-larva")
        assert "Chemical" in rec
        assert "Apply" in rec
        assert "|" in rec

    def test_format_recommendation_all_classes(self):
        """Should return non-empty string for all known classes."""
        for class_name in recommendations.RECOMMENDATION_RULES.keys():
            rec = recommendations.format_recommendation(class_name)
            assert isinstance(rec, str)
            assert len(rec) > 0

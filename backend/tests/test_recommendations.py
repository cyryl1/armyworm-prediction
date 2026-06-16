"""Tests for recommendation logic."""

from app import recommendations


class TestRecommendations:
    """Test pest management recommendation generation."""

    def test_get_recommendation_details_armyworm_larva(self):
        """Should return high severity for active larvae with full protocol."""
        details = recommendations.get_recommendation_details("fall-armyworm-larva")
        assert details["severity"] == "high"
        assert details["alert_color"] == "red"
        assert len(details["cultural_control"]) > 0
        assert len(details["biological_control"]) > 0
        assert len(details["chemical_control"]) > 0
        assert len(details["prevention"]) > 0
        assert len(details["sources"]) > 0

    def test_get_recommendation_details_healthy(self):
        """Should return no-action severity for healthy maize."""
        details = recommendations.get_recommendation_details("healthy-maize")
        assert details["severity"] == "none"
        assert details["alert_color"] == "green"

    def test_get_recommendation_details_unknown_class(self):
        """Should return sensible defaults for unknown class."""
        details = recommendations.get_recommendation_details("unknown-pest")
        assert details["severity"] == "unknown"
        assert "Unrecognised" in details["description"]

    def test_format_recommendation_compact(self):
        """Should format as compact one-liner."""
        rec = recommendations.format_recommendation("fall-armyworm-larva")
        assert "HIGH" in rec
        assert "|" in rec

    def test_format_recommendation_all_classes(self):
        """Should return non-empty string for all known classes."""
        for class_name in recommendations.MANAGEMENT_PROTOCOLS.keys():
            rec = recommendations.format_recommendation(class_name)
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_region_advisory_with_gps(self):
        """Should include region advisory when GPS is provided."""
        # Nairobi, Kenya
        details = recommendations.get_recommendation_details(
            "fall-armyworm-larva", latitude=-1.286, longitude=36.817
        )
        assert "region_advisory" in details
        assert details["region_advisory"]["region_name"] == "Sub-Saharan Africa"

    def test_region_advisory_default(self):
        """Should fall back to default region when no GPS is given."""
        details = recommendations.get_recommendation_details("fall-armyworm-larva")
        assert details["region_advisory"]["region_name"] == "General"

    def test_all_classes_have_sources(self):
        """Every known class should cite authoritative sources."""
        for class_name, protocol in recommendations.MANAGEMENT_PROTOCOLS.items():
            assert "sources" in protocol, f"Missing sources for {class_name}"
            assert len(protocol["sources"]) > 0, f"Empty sources for {class_name}"

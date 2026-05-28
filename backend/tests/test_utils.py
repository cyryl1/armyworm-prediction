"""Tests for utility functions (EXIF, annotation, file handling)."""

import pytest
from pathlib import Path
from PIL import Image
import cv2

from app import utils


class TestGPSExtraction:
    """Test EXIF GPS data extraction."""

    def test_extract_gps_from_image_without_exif(self, temp_image_jpg):
        """Should return (None, None) for images without EXIF data."""
        lat, lon = utils.extract_gps_from_exif(Path(temp_image_jpg))
        assert lat is None
        assert lon is None

    def test_extract_gps_handles_missing_file(self):
        """Should return (None, None) for non-existent files."""
        lat, lon = utils.extract_gps_from_exif(Path("/nonexistent/file.jpg"))
        assert lat is None
        assert lon is None


class TestAnnotation:
    """Test image annotation with detection boxes."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Ensure temp directory exists during annotation tests."""
        utils.init_temp_directory()
        yield
        utils.cleanup_temp_directory()

    def test_annotate_detections_creates_output(self, temp_image_jpg, sample_detections):
        """Should create an annotated image file."""
        output_path = utils.annotate_detections(Path(temp_image_jpg), sample_detections)
        assert output_path.exists()
        assert output_path.suffix == ".jpg"
        
        # Verify output can be read
        img = cv2.imread(str(output_path))
        assert img is not None

    def test_annotate_detections_with_custom_name(self, temp_image_jpg, sample_detections):
        """Should respect custom output filename."""
        custom_name = "test_annotated.jpg"
        output_path = utils.annotate_detections(Path(temp_image_jpg), sample_detections, output_filename=custom_name)
        assert custom_name in output_path.name

    def test_annotate_detections_invalid_image(self, sample_detections):
        """Should raise ValueError for non-existent image."""
        with pytest.raises(ValueError, match="Cannot read image"):
            utils.annotate_detections(Path("/nonexistent/image.jpg"), sample_detections)


class TestFileHandling:
    """Test file upload and storage utilities."""

    def test_validate_file_extension_allowed(self):
        """Should accept allowed image extensions."""
        assert utils.validate_file_extension("photo.jpg") is True
        assert utils.validate_file_extension("image.png") is True
        assert utils.validate_file_extension("scan.jpeg") is True

    def test_validate_file_extension_rejected(self):
        """Should reject non-image files."""
        assert utils.validate_file_extension("document.txt") is False
        assert utils.validate_file_extension("archive.zip") is False

    def test_validate_file_size_within_limit(self):
        """Should accept files within size limit."""
        assert utils.validate_file_size(1024 * 1024) is True  # 1MB
        assert utils.validate_file_size(5 * 1024 * 1024) is True  # 5MB

    def test_validate_file_size_exceeds_limit(self):
        """Should reject files exceeding limit."""
        assert utils.validate_file_size(11 * 1024 * 1024) is False  # 11MB > 10MB default

    def test_generate_unique_filename(self):
        """Should generate unique filenames with UUID prefix."""
        name1 = utils.generate_unique_filename("test.jpg")
        name2 = utils.generate_unique_filename("test.jpg")
        assert name1 != name2
        assert name1.endswith("test.jpg")
        assert "_" in name1  # Contains UUID


class TestTempDirectory:
    """Test temporary directory management."""

    def test_init_temp_directory_creates_dir(self):
        """Should create temp directory if it doesn't exist."""
        utils.init_temp_directory()
        assert utils.TEMP_DIR.exists()

    def test_get_temp_file_path(self):
        """Should return valid temp file path."""
        path = utils.get_temp_file_path("test.jpg")
        assert str(path).startswith(str(utils.TEMP_DIR))
        assert path.name == "test.jpg"

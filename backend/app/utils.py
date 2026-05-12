"""
Utility functions for file handling and image processing.

This module provides helpers for managing uploaded files including
validation, temporary storage, and cleanup operations.
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Temporary directory for storing uploaded files
TEMP_DIR = Path("./temp")


def init_temp_directory() -> None:
    """Initialize temporary directory for file uploads.
    
    Creates temp directory if it doesn't exist.
    """
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info(f"Initialized temp directory: {TEMP_DIR}")


def validate_file_extension(filename: str) -> bool:
    """Validate if file has allowed image extension.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    is_valid = file_ext in ALLOWED_EXTENSIONS
    
    if not is_valid:
        logger.warning(f"Invalid file extension: {file_ext}")
    
    return is_valid


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate if file size is within limits.
    
    Args:
        file_size: Size of file in bytes
        max_size_mb: Maximum allowed size in megabytes
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    max_bytes = max_size_mb * 1024 * 1024
    is_valid = file_size <= max_bytes
    
    if not is_valid:
        logger.warning(f"File size {file_size} bytes exceeds limit {max_bytes}")
    
    return is_valid


def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename to avoid collisions.
    
    Args:
        original_filename: Original name of the uploaded file
        
    Returns:
        Unique filename with UUID prefix
    """
    file_ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    unique_filename = f"{unique_id}_{original_filename}"
    
    return unique_filename


def get_temp_file_path(filename: str) -> Path:
    """Get full path for temporary file.
    
    Args:
        filename: Name of the temporary file
        
    Returns:
        Full path to temporary file
    """
    return TEMP_DIR / filename


def save_uploaded_file(file_content: bytes, original_filename: str) -> Tuple[str, Path]:
    """Save uploaded file to temporary directory.
    
    Args:
        file_content: Raw bytes of the uploaded file
        original_filename: Original filename from upload
        
    Returns:
        Tuple of (unique_filename, full_file_path)
        
    Raises:
        ValueError: If file validation fails
    """
    # Validate extension
    if not validate_file_extension(original_filename):
        raise ValueError(f"Invalid file type: {original_filename}")
    
    # Validate size
    if not validate_file_size(len(file_content)):
        raise ValueError(f"File size exceeds maximum limit")
    
    # Generate unique filename
    unique_filename = generate_unique_filename(original_filename)
    file_path = get_temp_file_path(unique_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved uploaded file: {file_path}")
        return unique_filename, file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise


def delete_file(file_path: Path) -> bool:
    """Delete temporary file after processing.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def cleanup_temp_directory() -> None:
    """Clean up temporary directory by removing all files.
    
    Useful for cleanup on shutdown or maintenance.
    """
    try:
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                if file.is_file():
                    os.remove(file)
            logger.info("Cleaned up temp directory")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {e}")

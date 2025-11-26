"""
Dataset loading module for MAIRA-2 evaluation.

This module provides classes and functions to load evaluation datasets
from various formats (CSV, DICOM, images) for radiology report evaluation.
"""
import os
import re
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
from html import unescape

from PIL import Image

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


@dataclass
class DatasetItem:
    """
    Represents a single evaluation sample with image and report data.
    
    Attributes:
        instance_id: Unique identifier for the sample
        image_path: Path to the image file (JPG or DICOM)
        reference_report: Ground truth radiology report
        indication: Clinical indication/reason for exam (optional)
        technique: Imaging technique description (optional)
        comparison: Prior study comparison text (optional)
        metadata: Additional metadata from the dataset
    """
    instance_id: str
    image_path: str
    reference_report: str
    indication: Optional[str] = None
    technique: Optional[str] = None
    comparison: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def load_image(self) -> Image.Image:
        """
        Load the image from disk.
        
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        ext = Path(self.image_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            return Image.open(self.image_path).convert('RGB')
        elif ext in ['.dcm', '.dicom']:
            if not PYDICOM_AVAILABLE:
                raise ImportError(
                    "pydicom is required to load DICOM files. "
                    "Install with: pip install pydicom"
                )
            return self._load_dicom()
        else:
            raise ValueError(f"Unsupported image format: {ext}")
    
    def _load_dicom(self) -> Image.Image:
        """Load and convert DICOM file to PIL Image."""
        import numpy as np
        
        ds = pydicom.dcmread(self.image_path)
        pixel_array = ds.pixel_array
        
        # Normalize to 0-255 range
        if pixel_array.dtype != np.uint8:
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Handle photometric interpretation (invert if needed)
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = 255 - pixel_array
        
        # Convert to RGB
        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            image = Image.fromarray(pixel_array).convert('RGB')
        
        return image


def clean_html_report(html_text: str) -> str:
    """
    Clean HTML-formatted radiology report to plain text.
    
    Args:
        html_text: HTML-formatted report text
        
    Returns:
        Cleaned plain text report
    """
    if not html_text:
        return ""
    
    # Unescape HTML entities
    text = unescape(html_text)
    
    # Replace common HTML tags with appropriate spacing
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = text.strip()
    
    return text


class EvaluationDataset:
    """
    Dataset class for loading and iterating over evaluation samples.
    
    Supports loading from CSV files with associated image directories.
    Can use either JPG images or DICOM files as the image source.
    
    Example:
        ```python
        dataset = EvaluationDataset(
            csv_path="data/labels.csv",
            images_dir="data/images/",
            image_format="jpg"
        )
        
        for item in dataset:
            image = item.load_image()
            reference = item.reference_report
            # Process item...
        ```
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        image_format: str = "jpg",
        instance_id_column: str = "instance_id",
        report_column: str = "report_result",
        indication_column: Optional[str] = None,
        technique_column: Optional[str] = None,
        comparison_column: Optional[str] = None,
        clean_html: bool = True,
        filter_empty_reports: bool = True,
    ):
        """
        Initialize the evaluation dataset.
        
        Args:
            csv_path: Path to the CSV file containing metadata
            images_dir: Directory containing image files
            image_format: Image format to use ('jpg', 'png', 'dcm')
            instance_id_column: CSV column name for instance IDs
            report_column: CSV column name for reference reports
            indication_column: CSV column name for clinical indication
            technique_column: CSV column name for technique description
            comparison_column: CSV column name for comparison text
            clean_html: Whether to clean HTML from reports
            filter_empty_reports: Whether to skip entries with empty reports
        """
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.image_format = image_format.lower().lstrip('.')
        self.instance_id_column = instance_id_column
        self.report_column = report_column
        self.indication_column = indication_column
        self.technique_column = technique_column
        self.comparison_column = comparison_column
        self.clean_html = clean_html
        self.filter_empty_reports = filter_empty_reports
        
        self._items: List[DatasetItem] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and parse the CSV dataset."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                instance_id = row.get(self.instance_id_column)
                if not instance_id:
                    continue
                
                # Get report text
                report = row.get(self.report_column, "")
                if self.clean_html:
                    report = clean_html_report(report)
                
                if self.filter_empty_reports and not report.strip():
                    continue
                
                # Build image path
                image_path = self.images_dir / f"{instance_id}.{self.image_format}"
                if not image_path.exists():
                    # Try without extension cleanup
                    possible_files = list(self.images_dir.glob(f"{instance_id}.*"))
                    if possible_files:
                        image_path = possible_files[0]
                    else:
                        continue  # Skip if no image found
                
                # Get optional fields
                indication = None
                if self.indication_column:
                    indication = row.get(self.indication_column)
                    if indication and self.clean_html:
                        indication = clean_html_report(indication)
                
                technique = None
                if self.technique_column:
                    technique = row.get(self.technique_column)
                    if technique and self.clean_html:
                        technique = clean_html_report(technique)
                
                comparison = None
                if self.comparison_column:
                    comparison = row.get(self.comparison_column)
                    if comparison and self.clean_html:
                        comparison = clean_html_report(comparison)
                
                # Store all other columns as metadata
                metadata = {k: v for k, v in row.items() if k not in [
                    self.instance_id_column, self.report_column,
                    self.indication_column, self.technique_column,
                    self.comparison_column
                ]}
                
                item = DatasetItem(
                    instance_id=instance_id,
                    image_path=str(image_path),
                    reference_report=report,
                    indication=indication,
                    technique=technique,
                    comparison=comparison,
                    metadata=metadata,
                )
                
                self._items.append(item)
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)
    
    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over dataset items."""
        return iter(self._items)
    
    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item by index."""
        return self._items[idx]
    
    def get_by_id(self, instance_id: str) -> Optional[DatasetItem]:
        """Get item by instance ID."""
        for item in self._items:
            if item.instance_id == instance_id:
                return item
        return None
    
    def get_all_references(self) -> List[str]:
        """Get all reference reports."""
        return [item.reference_report for item in self._items]
    
    def get_all_instance_ids(self) -> List[str]:
        """Get all instance IDs."""
        return [item.instance_id for item in self._items]


def load_evaluation_dataset(
    data_dir: str,
    csv_filename: str = "merged_june_andres_labeling.csv",
    images_subdir: str = "images",
    **kwargs
) -> EvaluationDataset:
    """
    Convenience function to load an evaluation dataset.
    
    Args:
        data_dir: Base directory containing CSV and images
        csv_filename: Name of the CSV file
        images_subdir: Subdirectory containing images
        **kwargs: Additional arguments passed to EvaluationDataset
        
    Returns:
        Loaded EvaluationDataset instance
    """
    data_path = Path(data_dir)
    csv_path = data_path / csv_filename
    images_dir = data_path / images_subdir
    
    return EvaluationDataset(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        **kwargs
    )

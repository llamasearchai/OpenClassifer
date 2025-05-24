"""
Data utility functions for OpenClassifier.
Provides data loading, export, and import functionality.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import asyncio
import aiofiles
from datetime import datetime
import pickle
import numpy as np

from open_classifier.core.logging import get_logger
from open_classifier.core.config import settings

logger = get_logger(__name__)


class DataLoader:
    """
    Comprehensive data loading utility for various formats.
    Supports CSV, JSON, Excel, and streaming for large datasets.
    """

    def __init__(self, batch_size: int = 1000):
        """
        Initialize DataLoader.
        
        Args:
            batch_size: Default batch size for streaming operations
        """
        self.batch_size = batch_size

    def load_csv(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = "label",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of the text column
            label_column: Name of the label column (optional)
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            if label_column and label_column not in df.columns:
                logger.warning(f"Label column '{label_column}' not found in CSV")
                label_column = None
            
            # Clean and validate data
            df = self._clean_dataframe(df, text_column, label_column)
            
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def load_json(
        self,
        file_path: Union[str, Path],
        text_field: str = "text",
        label_field: Optional[str] = "label"
    ) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Name of the text field
            label_field: Name of the label field (optional)
            
        Returns:
            List of data records
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                data = [data]
            
            # Validate and clean data
            cleaned_data = []
            for record in data:
                if text_field in record:
                    cleaned_record = {
                        "text": str(record[text_field]).strip(),
                        "label": record.get(label_field) if label_field else None
                    }
                    if cleaned_record["text"]:
                        cleaned_data.append(cleaned_record)
            
            logger.info(f"Loaded {len(cleaned_data)} records from {file_path}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise

    def load_excel(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = "label",
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path: Path to Excel file
            text_column: Name of the text column
            label_column: Name of the label column (optional)
            sheet_name: Name of the sheet to read (optional)
            **kwargs: Additional arguments for pandas.read_excel
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in Excel")
            
            if label_column and label_column not in df.columns:
                logger.warning(f"Label column '{label_column}' not found in Excel")
                label_column = None
            
            # Clean and validate data
            df = self._clean_dataframe(df, text_column, label_column)
            
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise

    def stream_csv(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = "label",
        batch_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Stream data from CSV file in batches.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of the text column
            label_column: Name of the label column (optional)
            batch_size: Batch size for streaming
            
        Yields:
            DataFrame batches
        """
        batch_size = batch_size or self.batch_size
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=batch_size):
                # Validate required columns
                if text_column not in chunk.columns:
                    raise ValueError(f"Text column '{text_column}' not found in CSV")
                
                # Clean and validate chunk
                chunk = self._clean_dataframe(chunk, text_column, label_column)
                
                if not chunk.empty:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error streaming CSV file {file_path}: {e}")
            raise

    def _clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Clean and validate DataFrame."""
        # Remove rows with empty text
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != ""]
        
        # Clean text column
        df[text_column] = df[text_column].astype(str).str.strip()
        
        # Clean label column if present
        if label_column and label_column in df.columns:
            df = df.dropna(subset=[label_column])
            df[label_column] = df[label_column].astype(str).str.strip()
        
        return df

    async def load_json_async(
        self,
        file_path: Union[str, Path],
        text_field: str = "text",
        label_field: Optional[str] = "label"
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Name of the text field
            label_field: Name of the label field (optional)
            
        Returns:
            List of data records
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                data = [data]
            
            # Validate and clean data
            cleaned_data = []
            for record in data:
                if text_field in record:
                    cleaned_record = {
                        "text": str(record[text_field]).strip(),
                        "label": record.get(label_field) if label_field else None
                    }
                    if cleaned_record["text"]:
                        cleaned_data.append(cleaned_record)
            
            logger.info(f"Loaded {len(cleaned_data)} records from {file_path}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise


def export_results(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "json",
    include_metadata: bool = True
) -> None:
    """
    Export classification results to various formats.
    
    Args:
        results: List of classification results
        output_path: Output file path
        format: Export format (json, csv, excel)
        include_metadata: Whether to include metadata in export
    """
    output_path = Path(output_path)
    
    try:
        if format.lower() == "json":
            _export_json(results, output_path, include_metadata)
        elif format.lower() == "csv":
            _export_csv(results, output_path, include_metadata)
        elif format.lower() in ["excel", "xlsx"]:
            _export_excel(results, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting results to {output_path}: {e}")
        raise


def _export_json(
    results: List[Dict[str, Any]],
    output_path: Path,
    include_metadata: bool
) -> None:
    """Export results to JSON format."""
    export_data = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "total_records": len(results),
            "format_version": "1.0"
        },
        "results": results if include_metadata else [
            {k: v for k, v in result.items() if k != "metadata"} 
            for result in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def _export_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    include_metadata: bool
) -> None:
    """Export results to CSV format."""
    if not results:
        return
    
    # Flatten nested dictionaries
    flattened_results = []
    for result in results:
        flat_result = {}
        
        # Basic fields
        flat_result["text"] = result.get("text", "")
        
        # Classification results
        if "classification" in result:
            classification = result["classification"]
            flat_result["predicted_class"] = classification.get("class", "")
            flat_result["confidence"] = classification.get("confidence", 0.0)
            
            # Probabilities
            if "probabilities" in classification:
                for label, prob in classification["probabilities"].items():
                    flat_result[f"prob_{label}"] = prob
        
        # Metadata (if requested)
        if include_metadata and "metadata" in result:
            metadata = result["metadata"]
            for key, value in metadata.items():
                flat_result[f"meta_{key}"] = value
        
        flattened_results.append(flat_result)
    
    # Create DataFrame and export
    df = pd.DataFrame(flattened_results)
    df.to_csv(output_path, index=False)


def _export_excel(
    results: List[Dict[str, Any]],
    output_path: Path,
    include_metadata: bool
) -> None:
    """Export results to Excel format."""
    if not results:
        return
    
    # Use the same flattening logic as CSV
    flattened_results = []
    for result in results:
        flat_result = {}
        
        # Basic fields
        flat_result["text"] = result.get("text", "")
        
        # Classification results
        if "classification" in result:
            classification = result["classification"]
            flat_result["predicted_class"] = classification.get("class", "")
            flat_result["confidence"] = classification.get("confidence", 0.0)
            
            # Probabilities
            if "probabilities" in classification:
                for label, prob in classification["probabilities"].items():
                    flat_result[f"prob_{label}"] = prob
        
        # Metadata (if requested)
        if include_metadata and "metadata" in result:
            metadata = result["metadata"]
            for key, value in metadata.items():
                flat_result[f"meta_{key}"] = value
        
        flattened_results.append(flat_result)
    
    # Create DataFrame and export with formatting
    df = pd.DataFrame(flattened_results)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Format the worksheet
        worksheet = writer.sheets['Results']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width


def import_data(
    file_path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Import data from various formats with automatic format detection.
    
    Args:
        file_path: Path to data file
        format: File format (auto-detected if None)
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        Loaded data in appropriate format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format if not specified
    if format is None:
        format = file_path.suffix.lower().lstrip('.')
    
    loader = DataLoader()
    
    try:
        if format in ['csv']:
            return loader.load_csv(file_path, **kwargs)
        elif format in ['json', 'jsonl']:
            return loader.load_json(file_path, **kwargs)
        elif format in ['xlsx', 'xls', 'excel']:
            return loader.load_excel(file_path, **kwargs)
        elif format in ['pkl', 'pickle']:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {format}")
            
    except Exception as e:
        logger.error(f"Error importing data from {file_path}: {e}")
        raise


def save_embeddings(
    embeddings: np.ndarray,
    texts: List[str],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save embeddings with associated texts and metadata.
    
    Args:
        embeddings: Numpy array of embeddings
        texts: List of corresponding texts
        output_path: Output file path
        metadata: Optional metadata dictionary
    """
    output_path = Path(output_path)
    
    save_data = {
        "embeddings": embeddings.tolist(),
        "texts": texts,
        "metadata": metadata or {},
        "shape": embeddings.shape,
        "dtype": str(embeddings.dtype),
        "created_at": datetime.now().isoformat()
    }
    
    try:
        if output_path.suffix.lower() == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving embeddings to {output_path}: {e}")
        raise


def load_embeddings(
    file_path: Union[str, Path]
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load embeddings with associated texts and metadata.
    
    Args:
        file_path: Path to embeddings file
        
    Returns:
        Tuple of (embeddings array, texts list, metadata dict)
    """
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        embeddings = np.array(data["embeddings"])
        texts = data["texts"]
        metadata = data.get("metadata", {})
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
        return embeddings, texts, metadata
        
    except Exception as e:
        logger.error(f"Error loading embeddings from {file_path}: {e}")
        raise


def create_train_test_split(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True
) -> tuple:
    """
    Create train/test split for classification data.
    
    Args:
        data: Input data
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by labels
        
    Returns:
        Tuple of (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    if stratify and "label" in df.columns:
        stratify_column = df["label"]
    else:
        stratify_column = None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_column
    )
    
    logger.info(f"Created train/test split: {len(train_df)} train, {len(test_df)} test")
    
    if isinstance(data, list):
        return train_df.to_dict('records'), test_df.to_dict('records')
    else:
        return train_df, test_df 
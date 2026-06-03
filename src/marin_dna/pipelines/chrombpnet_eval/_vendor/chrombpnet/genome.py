from __future__ import annotations
from collections.abc import Callable

from pathlib import Path
from pooch import Decompress
import pandas as pd
import os
import logging

import pooch

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedDataset:
    """
    Enhanced dataset class that provides better caching and error handling.
    """
    
    def __init__(self, dataset_func):
        self.dataset_func = dataset_func
        self._dataset = None
    
    @property
    def dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            self._dataset = self.dataset_func()
        return self._dataset
    
    def check_file_exists(self, filename):
        """
        Check if a file is already downloaded and valid.
        
        Parameters
        ----------
        filename : str
            Name of the file to check
            
        Returns
        -------
        bool
            True if file exists and is valid, False otherwise
        """
        try:
            file_path = self.dataset.abspath / filename
            return file_path.exists() and self.dataset.is_available(filename)
        except Exception as e:
            logger.warning(f"Error checking file {filename}: {e}")
            return False
    
    def fetch(self, filename, processor=None, progressbar=True):
        """
        Safely fetch a file with better error handling and logging.
        
        Parameters
        ----------
        filename : str
            Name of the file to fetch
        processor : callable, optional
            Processor to apply to the downloaded file
        progressbar : bool
            Whether to show progress bar
            
        Returns
        -------
        str
            Path to the downloaded file
        """
        # Check if file already exists
        if self.check_file_exists(filename):
            logger.info(f"File {filename} already exists and is valid, skipping download")
            return str(self.dataset.abspath / filename)
        
        logger.info(f"Downloading {filename}...")
        try:
            result = self.dataset.fetch(filename, processor=processor, progressbar=progressbar)
            logger.info(f"Successfully downloaded {filename}")
            return result
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    
    def get_cache_info(self):
        """
        Get information about the cache directory and available files.
        
        Returns
        -------
        dict
            Dictionary with cache information
        """
        info = {
            'cache_directory': str(self.dataset.abspath),
            'base_url': self.dataset.base_url,
            'available_files': list(self.dataset.registry.keys()),
            'downloaded_files': []
        }
        
        for filename in self.dataset.registry.keys():
            if self.check_file_exists(filename):
                file_path = self.dataset.abspath / filename
                file_size = file_path.stat().st_size if file_path.exists() else 0
                info['downloaded_files'].append({
                    'filename': filename,
                    'size_bytes': file_size,
                    'size_mb': file_size / (1024**2)
                })
        
        return info

# Legacy functions for backward compatibility
def check_file_exists(dataset_func, filename):
    """Legacy function - use EnhancedDataset.check_file_exists instead."""
    dataset = EnhancedDataset(dataset_func)
    return dataset.check_file_exists(filename)

def safe_fetch(dataset_func, filename, processor=None, progressbar=True):
    """Legacy function - use EnhancedDataset.safe_fetch instead."""
    dataset = EnhancedDataset(dataset_func)
    return dataset.safe_fetch(filename, processor, progressbar)

def motifs_datasets():
    """Get enhanced motifs dataset."""
    def _create_dataset():
        return pooch.create(
            path=pooch.os_cache("genome/motifs"),
            base_url="https://zenodo.org/records/7445373/files/",
            env="GENOME_DATA_DIR",
            registry={
                "motifs.meme.txt": "md5:21b0d5f5496efe677567db04304ed0de",
            },
            urls={
                "motifs.meme.txt": "https://zenodo.org/records/7445373/files/motifs.meme.txt?download=1",
            },
        )
    return EnhancedDataset(_create_dataset)

def hg38_datasets():
    """Get enhanced hg38 dataset."""
    def _create_dataset():
        return pooch.create(
                path=pooch.os_cache("genome/hg38"),
                base_url="https://zenodo.org/records/12193595/files/",
                env="GENOME_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
                # The registry specifies the files that can be fetched
                registry={
                    # TF motifs
                    # Genome files
                    "hg38.chrom.sizes": "md5:c95303fb77cc3e11d50e3c3a4b93b3fb",
                    "hg38.fa": "md5:a6da8681616c05eb542f1d91606a7b2f",
                    "hg38.gtf.gz": "md5:16fcae8ca8e488cd8056cf317d963407",
                    "fold_0.json": "md5:88bab8abe271c9ebb6655a0332b74998",
                    "fold_1.json": "md5:426f117b2d4e5885fb10ef7a3b7e593e",
                    "fold_2.json": "md5:b603378ebfcd8954aecd4ae60c4ce9b4",
                    "fold_3.json": "md5:8e70574ae38b7314c09cfc7db6194486",
                    "fold_4.json": "md5:eab6e147532e3cb5c8e6860b2e24da3b",
                },
                urls={
                    "hg38.fa": "https://zenodo.org/records/12193595/files/hg38.fa",
                    "hg38.gtf.gz": "https://zenodo.org/records/12193595/files/gencode.v38.annotation.gtf.gz",
                    "hg38.chrom.sizes": "https://zenodo.org/records/12193595/files/hg38.chrom.sizes",
                    "fold_0.json": "https://zenodo.org/records/12193595/files/fold_0.json",
                    "fold_1.json": "https://zenodo.org/records/12193595/files/fold_1.json",
                    "fold_2.json": "https://zenodo.org/records/12193595/files/fold_2.json",
                    "fold_3.json": "https://zenodo.org/records/12193595/files/fold_3.json",
                    "fold_4.json": "https://zenodo.org/records/12193595/files/fold_4.json",
                },
            )
    return EnhancedDataset(_create_dataset)

def mm10_datasets():
    """Get enhanced mm10 dataset."""
    def _create_dataset():
        return pooch.create(
            path=pooch.os_cache("genome/mm10"),
            base_url="https://zenodo.org/records/12193429/files/",
            env="GENOME_DATA_DIR",
            registry={
                "mm10.gtf.gz": "md5:5a103c9a15dd660c295a089ef5035672",
                "mm10.fa": "md5:7e329b2bf419a9f5a7dc42c625c884ac",
                "mm10.chrom.sizes": "md5:5a103c9a15dd660c295a089ef5035672",
                "fold_0.json": "md5:32f25d53ba1270fe2acd15b31bc4789d",
                "fold_1.json": "md5:b07be1148b1e81399436c985c0647999",
                "fold_2.json": "md5:fccbd2d245aca6433ad6f729a2e5d9a8",
                "fold_3.json": "md5:c74f9221fc7f0d5984368c4cf594a071",
                "fold_4.json": "md5:f3eaf62e8a013db9de19f606a46dba04",
            },
            urls={
                "mm10.gtf.gz": "https://zenodo.org/records/12193429/files/mm10.gtf.gz",
                "mm10.fa": "https://zenodo.org/records/12193429/files/mm10.fa",
                "mm10.chrom.sizes": "https://zenodo.org/records/12193429/files/mm10.chrom.sizes",
                "fold_0.json": "https://zenodo.org/records/12193429/files/fold_0.json",
                "fold_1.json": "https://zenodo.org/records/12193429/files/fold_1.json",
                "fold_2.json": "https://zenodo.org/records/12193429/files/fold_2.json",
                "fold_3.json": "https://zenodo.org/records/12193429/files/fold_3.json",
                "fold_4.json": "https://zenodo.org/records/12193429/files/fold_4.json",
            },
        )
    return EnhancedDataset(_create_dataset)

def hg19_datasets():
    """Get enhanced hg19 dataset."""
    def _create_dataset():
        return pooch.create(
            path=pooch.os_cache("genome/hg19"),
            base_url="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/",
            env="GENOME_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
            registry={
                "hg19.fa.gz": "md5:806c02398f5ac5da8ffd6da2d1d5d1a9",
                "hg19.gtf.gz": "md5:bd83e28270e595d3bde6bfcb21c9748f",
                "hg19.chrom.sizes": "md5:b3b0fcf79b5477ab0b3af02e81eac8dc",
                "hg19.fa": "md5:530d89d3ef07fdb2a9b3c701fb4ca486",
                "fold_0.json": "md5:88bab8abe271c9ebb6655a0332b74998",
                "fold_1.json": "md5:426f117b2d4e5885fb10ef7a3b7e593e",
                "fold_2.json": "md5:b603378ebfcd8954aecd4ae60c4ce9b4",
                "fold_3.json": "md5:8e70574ae38b7314c09cfc7db6194486",
                "fold_4.json": "md5:eab6e147532e3cb5c8e6860b2e24da3b",
            },
            urls={
                "hg19.fa": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
                "hg19.gtf.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz",
                "hg19.chrom.sizes": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes",
                "fold_0.json": "https://zenodo.org/records/12193595/files/fold_0.json",
                "fold_1.json": "https://zenodo.org/records/12193595/files/fold_1.json",
                "fold_2.json": "https://zenodo.org/records/12193595/files/fold_2.json",
                "fold_3.json": "https://zenodo.org/records/12193595/files/fold_3.json",
                "fold_4.json": "https://zenodo.org/records/12193595/files/fold_4.json",
            },
        )
    return EnhancedDataset(_create_dataset)

class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence,
    its annotation, and chromosome sizes.

    Attributes
    ----------
    fasta
        The path to the FASTA file.
    annotation
        The path to the annotation file.
    chrom_sizes
        A dictionary containing chromosome names and sizes.

    Raises
    ------
    ValueError
        If `fasta` or `annotation` are not a Path, a string, or a callable.
    """

    def __init__(
        self,
        *,
        fasta: Path | Callable[[], Path], 
        annotation: Path | Callable[[], Path],
        chrom_sizes: Path | Callable[[], Path],
        # chrom_sizes: dict[str, int] | None = None,
    ):
        """
        Initializes the Genome object with paths or callables for FASTA and annotation files,
        and optionally, chromosome sizes.

        Parameters
        ----------
        fasta
            A Path or callable that returns a Path to the FASTA file.
        annotation
            A Path or callable that returns a Path to the annotation file.
        chrom_sizes
            Optional chromosome sizes. If not provided, chromosome sizes will
            be inferred from the FASTA file.
        """
        if callable(fasta):
            self._fetch_fasta = fasta
            self._fasta = None
        elif isinstance(fasta, Path) or isinstance(fasta, str):
            self._fasta = Path(fasta)
            self._fetch_fasta = None
        else:
            raise ValueError("fasta must be a Path or Callable")

        if callable(annotation):
            self._fetch_annotation = annotation
            self._annotation = None
        elif isinstance(annotation, Path) or isinstance(annotation, str):
            self._annotation = Path(annotation)
            self._fetch_annotation = None
        else:
            raise ValueError("annotation must be a Path or Callable")

        self._chrom_sizes = chrom_sizes
        self.chrom_sizes = chrom_sizes

    @property
    def fasta(self):
        """
        The Path to the FASTA file. 

        Returns
        -------
        Path
            The path to the FASTA file.
        """
        if self._fasta is None:
            self._fasta = Path(self._fetch_fasta())
        return str(self._fasta)

    @property
    def annotation(self):
        """
        The Path to the annotation file.

        Returns
        -------
        Path
            The path to the annotation file.
        """
        if self._annotation is None:
            self._annotation = Path(self._fetch_annotation())
        return str(self._annotation)

    # @property
    # def chrom_sizes(self):
    #     """
    #     A dictionary with chromosome names as keys and their lengths as values.

    #     Returns
    #     -------
    #     dict[str, int]
    #         A dictionary of chromosome sizes.
    #     """
    #     if self._chrom_sizes is None:
    #         from pyfaidx import Fasta
    #         fasta = Fasta(self.fasta)
    #         self._chrom_sizes = {chr: len(fasta[chr]) for chr in fasta.keys()}
    #     return self._chrom_sizes
        

GRCh38 = Genome(
    fasta=lambda : hg38_datasets().fetch(
        "hg38.fa", 
        # processor=Decompress(method="gzip", name="hg38.fa"), 
        progressbar=True
    ),
    annotation=lambda : hg38_datasets().fetch(
        "hg38.gtf.gz", 
        progressbar=True
    ),
    chrom_sizes= hg38_datasets().fetch(
        "hg38.chrom.sizes", 
        progressbar=True
    ),
    )
hg38 = GRCh38

GRCh37 = Genome(
    fasta=lambda : hg19_datasets().fetch(
        "hg19.fa", 
        progressbar=True, processor=Decompress(method="gzip", name="hg19.fa")
    ),
    annotation=lambda : hg19_datasets().fetch(
        "hg19.gtf.gz", 
        progressbar=True
    ),
    chrom_sizes= hg19_datasets().fetch(
        "hg19.chrom.sizes", 
        progressbar=True
    ),
)
hg19 = GRCh37

GRCm38 = Genome(
    fasta=lambda : mm10_datasets().fetch(
        "mm10.fa", 
        # processor=Decompress(method="gzip", name="mm10.fa"), 
        progressbar=True
    ),
    annotation=lambda : mm10_datasets().fetch(
        "mm10.gtf.gz", 
        progressbar=True
    ),
    chrom_sizes= mm10_datasets().fetch(
        "mm10.chrom.sizes", 
        progressbar=True
    ),

    )
mm10 = GRCm38


def strand_specific_start_site(df):
    df = df.copy()
    if set(df["Strand"]) != set(["+", "-"]):
        raise ValueError("Not all features are strand specific!")

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df
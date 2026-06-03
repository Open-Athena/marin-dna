# Author: Lei Xiong <jsxlei@gmail.com>

"""
Data configuration classes for the RegNet project.

This module provides configuration classes for different data types and processing steps.
It includes parameter validation and default values for common configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from argparse import ArgumentParser, Namespace
import os
import json
from .parse_utils import add_argparse_args, from_argparse_args, parse_argparser, get_init_arguments_and_types
from .genome import hg38, hg38_datasets, mm10, mm10_datasets


class DataConfig:
	"""Base configuration class for data handling.
	
	This class defines the common parameters used across different data types
	and provides validation for these parameters.
	"""
	
	def __init__(
			self,
			data_dir: str = None,
			peaks: str = None, #'{}/peaks.bed',
			negatives: str = None, #'{}/negatives.bed', 
			bigwig: str = None, #'{}/unstranded.bw', 
			# background: str = None,
			negative_sampling_ratio: float = 0.1,
			saved_data: str = None,
			fasta: str = None,
			chrom_sizes: str = None,
			in_window: int = 2114,
			out_window: int = 1000,
			shift: int = 500,
			rc: float = 0.5,
			outlier_threshold: float = 0.999,
			data_type: str = 'profile',
			training_chroms: List = [
				"chr1",
				"chr2",
				"chr3",
				"chr4",
				"chr7",
				"chr8",
				"chr9",
				"chr11",
				"chr12",
				"chr13",
				"chr15",
				"chr16",
				"chr17",
				"chr19",
				"chrX",
				"chrY"
			],
			validation_chroms: List = ['chr6', 'chr21'],
			test_chroms: List = [
				"chr5",
				"chr10",
				"chr14",
				"chr18",
				"chr20",
				"chr22"
			],
			exclude_chroms: list = [],
			fold: int = 0,
			genome: str = 'hg38',
			batch_size: int = 64,
			num_workers: int = 32,
			debug: bool = False,
			**kwargs,
		):

			_genome = hg38 if genome == 'hg38' else mm10 if genome == 'mm10' else None
			_datasets = hg38_datasets() if genome == 'hg38' else mm10_datasets() if genome == 'mm10' else None
			self.data_dir = data_dir
			self.peaks = peaks if peaks is not None else f'{data_dir}/peaks.bed'
			self.negatives = negatives if negatives is not None else f'{data_dir}/negatives.bed' if data_dir is not None else None 
			self.bigwig = bigwig if bigwig is not None else f'{data_dir}/unstranded.bw'
			# self.background = background if background is not None else f'{data_dir}/background.bw'
			self.negative_sampling_ratio = negative_sampling_ratio
			self.saved_data = saved_data
			self.fasta = fasta if fasta is not None else _genome.fasta
			self.chrom_sizes = chrom_sizes if chrom_sizes is not None else _genome.chrom_sizes
			self.in_window = in_window
			self.out_window = out_window
			self.shift = shift
			self.rc = rc
			self.outlier_threshold = outlier_threshold
			self.data_type = data_type
			self.training_chroms = training_chroms
			self.validation_chroms = validation_chroms
			self.test_chroms = test_chroms
			self.exclude_chroms = exclude_chroms
			self.fold = fold
			self.batch_size = batch_size    
			self.num_workers = num_workers
			self.debug = debug

			fold_file_path = _datasets.fetch(f'fold_{fold}.json', progressbar=False)
			splits_dict = json.load(open(fold_file_path))
			# training_chroms, validation_chroms, test_chroms = None, None, None
			self.training_chroms = splits_dict['train'] if training_chroms is None else training_chroms
			self.validation_chroms = splits_dict['valid'] if validation_chroms is None else validation_chroms
			self.test_chroms = splits_dict['test'] if test_chroms is None else test_chroms
			print(self.test_chroms)
	
	def __post_init__(self):
		"""Validate configuration parameters after initialization."""
		self._validate_paths()
		self._validate_windows()
		self._validate_chromosomes()
		self._validate_data_type()
	
	def _validate_paths(self):
		"""Validate that all required files exist."""
		required_files = {
			'FASTA': self.fasta,
			'BigWig': self.bigwig,
			'Peaks': self.peaks
		}
		
		for name, path in required_files.items():
			if not os.path.exists(path):
				raise FileNotFoundError(f"{name} file not found: {path}")
	
	def _validate_windows(self):
		"""Validate window size parameters."""
		if self.in_window <= 0:
			raise ValueError("Input window size must be positive")
		if self.out_window <= 0:
			raise ValueError("Output window size must be positive")
		if self.in_window < self.out_window:
			raise ValueError("Input window must be larger than output window")
	
	def _validate_chromosomes(self):
		"""Validate chromosome configuration."""
		all_chroms = set(self.training_chroms + self.validation_chroms + self.test_chroms)
		excluded = set(self.exclude_chroms)
		
		if not all_chroms:
			raise ValueError("No chromosomes specified for training, validation, or testing")
		
		if excluded.intersection(all_chroms) != excluded:
			raise ValueError("Some excluded chromosomes are not in the training/validation/test sets")
	
	def _validate_data_type(self):
		"""Validate data type parameter."""
		if self.data_type not in ['profile', 'longrange']:
			raise ValueError("Data type must be either 'profile' or 'longrange'")
		
	@classmethod
	def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any):
		"""Extends existing argparse by default `LightningDataModule` attributes.

		Example::

			parser = ArgumentParser(add_help=False)
			parser = LightningDataModule.add_argparse_args(parser)
		"""
		return add_argparse_args(cls, parent_parser, **kwargs)


	@classmethod
	def from_argparse_args(
		cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
	) -> Union["pl.LightningDataModule", "pl.Trainer"]:
		"""Create an instance from CLI arguments.

		Args:
			args: The parser or namespace to take arguments from. Only known arguments will be
				parsed and passed to the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
			**kwargs: Additional keyword arguments that may override ones in the parser or namespace.
				These must be valid DataModule arguments.

		Example::

			module = LightningDataModule.from_argparse_args(args)
		"""
		return from_argparse_args(cls, args, **kwargs)


	@classmethod
	def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
		return parse_argparser(cls, arg_parser)

	@classmethod
	def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
		r"""Scans the DataModule signature and returns argument names, types and default values.

		Returns:
			List with tuples of 3 values:
			(argument name, set with argument types, argument default value).
		"""
		return get_init_arguments_and_types(cls)
	



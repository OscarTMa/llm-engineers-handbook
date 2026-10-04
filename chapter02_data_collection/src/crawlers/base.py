"""
Base Crawler abstract class.
"""

from abc import ABC, abstractmethod
from typing import List
from chapter02_data_collection.src.models import RawDocument


class BaseCrawler(ABC):

    @abstractmethod
    def extract(self) -> List[RawDocument]:
        """Extracts and normalizes raw records into RawDocument instances."""
        pass
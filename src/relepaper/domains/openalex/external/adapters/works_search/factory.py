from typing import Any, Dict

from relepaper.domains.openalex.external.adapters.works_search.works_search_adapter import PyAlexWorksSearchAdapter
from relepaper.domains.openalex.external.interfaces import IAdapter, IAdapterFactory


class OpenAlexWorksSearchAdapterFactory(IAdapterFactory):
    """Factory for creating OpenAlex adapters."""

    @staticmethod
    def create(strategy: str = "pyalex", config: Dict[str, Any] = None) -> IAdapter:
        """Create an OpenAlex adapter based on strategy.

        Args:
            strategy: adapter strategy ("pyalex", "requests", etc.)
            config: configuration for the adapter

        Returns:
            IAdapter instance
        """
        if config is None:
            config = {}

        match strategy:
            case "pyalex":
                return PyAlexWorksSearchAdapter(**config)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

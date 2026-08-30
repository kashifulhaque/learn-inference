"""Shared interface for the GPU backends that execute labs."""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


class ProviderError(RuntimeError):
    """The provider could not run the lab."""


class OutOfCredits(ProviderError):
    """The provider rejected the job because the account is out of credit.

    The app catches this to suggest switching to the fallback provider.
    """


@dataclass
class LabResult:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


class Provider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]:
        """Whether the provider is configured, and why not if it isn't."""

    def run_lab(
        self, lab_id: str, code: str, gpu: str, timeout: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events while the lab runs.

        Each event is a dict with a "type" of "log", "result", or "error".
        """

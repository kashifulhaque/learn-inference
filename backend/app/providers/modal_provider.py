"""Runs labs on Modal (https://modal.com).

The GPU code lives in gpu/modal_app.py and is deployed separately with
`modal deploy gpu/modal_app.py`. This module looks up the deployed function and
relays its output stream to the web app.
"""

import asyncio
import os
import queue
import re
import threading
from typing import Any, AsyncIterator

from ..config import get_settings
from .base import OutOfCredits, ProviderError

_CREDIT_PATTERNS = re.compile(
    r"out of credit|insufficient (?:credit|funds|balance)|quota exceeded|"
    r"payment required|billing|spending limit|no remaining credit",
    re.IGNORECASE,
)

_SENTINEL = object()


class ModalProvider:
    name = "modal"

    def __init__(self) -> None:
        self.settings = get_settings()

    def available(self) -> tuple[bool, str]:
        if not (self.settings.modal_token_id and self.settings.modal_token_secret):
            return False, "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are not set"
        return True, ""

    def _configure_env(self) -> None:
        os.environ.setdefault("MODAL_TOKEN_ID", self.settings.modal_token_id)
        os.environ.setdefault("MODAL_TOKEN_SECRET", self.settings.modal_token_secret)

    def _lookup(self, function_name: str):
        import modal

        self._configure_env()
        try:
            return modal.Function.from_name(self.settings.modal_app_name, function_name)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise ProviderError(
                f"Could not find the Modal function '{function_name}' in app "
                f"'{self.settings.modal_app_name}'. Deploy it with "
                f"`modal deploy gpu/modal_app.py`. ({exc})"
            ) from exc

    async def run_lab(
        self, lab_id: str, code: str, gpu: str, timeout: int
    ) -> AsyncIterator[dict[str, Any]]:
        # CPU-only labs go to the cheaper, faster-starting function.
        function_name = "run_lab_cpu" if gpu.lower() == "cpu" else "run_lab"
        fn = self._lookup(function_name)
        events: queue.Queue[Any] = queue.Queue(maxsize=1000)

        def pump() -> None:
            try:
                for event in fn.remote_gen(
                    lab_id=lab_id, code=code, gpu=gpu, timeout=timeout
                ):
                    events.put(event)
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                kind = "out_of_credits" if _CREDIT_PATTERNS.search(message) else "error"
                events.put({"type": kind, "message": message})
            finally:
                events.put(_SENTINEL)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        while True:
            event = await loop.run_in_executor(None, events.get)
            if event is _SENTINEL:
                break
            if isinstance(event, dict) and event.get("type") == "out_of_credits":
                raise OutOfCredits(event["message"])
            yield event

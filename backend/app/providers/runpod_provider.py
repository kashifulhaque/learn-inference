"""Fallback executor: a RunPod serverless endpoint.

Used when Modal credits run out. The worker image is in gpu/runpod_worker/ and
exposes the same lab contract as the Modal function. RunPod streams generator
output through /stream/{job_id}, which this provider polls.
"""

import asyncio
from typing import Any, AsyncIterator

import httpx

from ..config import get_settings
from .base import OutOfCredits, ProviderError

API_BASE = "https://api.runpod.ai/v2"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodProvider:
    name = "runpod"

    def __init__(self) -> None:
        self.settings = get_settings()

    def available(self) -> tuple[bool, str]:
        if not self.settings.runpod_api_key:
            return False, "RUNPOD_API_KEY is not set"
        if not self.settings.runpod_endpoint_id:
            return False, "RUNPOD_ENDPOINT_ID is not set"
        return True, ""

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.runpod_api_key}",
            "Content-Type": "application/json",
        }

    async def run_lab(
        self, lab_id: str, code: str, gpu: str, timeout: int
    ) -> AsyncIterator[dict[str, Any]]:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)

        endpoint = f"{API_BASE}/{self.settings.runpod_endpoint_id}"
        payload = {
            "input": {"lab_id": lab_id, "code": code, "gpu": gpu, "timeout": timeout}
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{endpoint}/run", json=payload, headers=self._headers
            )
            if response.status_code in (402, 403):
                raise OutOfCredits(f"RunPod rejected the job: {response.text}")
            response.raise_for_status()
            job_id = response.json()["id"]

            deadline = asyncio.get_running_loop().time() + timeout + 120
            while True:
                if asyncio.get_running_loop().time() > deadline:
                    raise ProviderError("RunPod job exceeded its deadline")

                stream = await client.get(
                    f"{endpoint}/stream/{job_id}", headers=self._headers
                )
                stream.raise_for_status()
                body = stream.json()

                for chunk in body.get("stream", []):
                    event = chunk.get("output")
                    if isinstance(event, dict):
                        yield event

                status = body.get("status")
                if status in TERMINAL:
                    if status != "COMPLETED":
                        yield {
                            "type": "error",
                            "message": f"RunPod job ended with status {status}",
                        }
                    return
                await asyncio.sleep(1.0)

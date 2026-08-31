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
from typing import Any, AsyncIterator, Callable, Coroutine

from ..config import get_settings
from .base import Entry, Instance, OutOfCredits, ProviderError, Volume

_CREDIT_PATTERNS = re.compile(
    r"out of credit|insufficient (?:credit|funds|balance)|quota exceeded|"
    r"payment required|billing|spending limit|no remaining credit",
    re.IGNORECASE,
)

_SENTINEL = object()

CONSOLE_URL = "https://modal.com/apps"


class ModalProvider:
    name = "modal"
    console_url = CONSOLE_URL

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

    # --- management ---------------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        """What is running on Modal, plus the volumes labs read weights from."""
        ok, why = self.available()
        if not ok:
            return {"available": False, "reason": why}

        self._configure_env()
        state = await _run_in_modal_loop(
            _collect,
            self.settings.modal_app_name,
            self.settings.modal_gpu,
        )

        notices = []
        if not state["deployed"]:
            notices.append(
                f"The app '{self.settings.modal_app_name}' is not deployed, so GPU "
                "labs cannot run on Modal. Deploy it with "
                "`modal deploy gpu/modal_app.py`."
            )
        return {
            "available": True,
            "reason": "",
            "instances": state["instances"],
            "volumes": state["volumes"],
            "notices": notices,
            "facts": [
                {"label": "Environments", "value": ", ".join(state["environments"])},
                {"label": "Lab app", "value": self.settings.modal_app_name},
                {
                    "label": "Deployed",
                    "value": "yes" if state["deployed"] else "no",
                },
                {"label": "GPU requested", "value": self.settings.modal_gpu},
            ],
            "functions": state["functions"],
            "apps": state["apps"],
        }

    async def act(self, action: str, target: str) -> str:
        """Stop a container or an ephemeral app."""
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        self._configure_env()

        if action == "stop-container":
            await _run_in_modal_loop(_stop_container, target)
            return f"Sent a stop signal to container {target}."
        if action == "stop-app":
            await _run_in_modal_loop(_stop_app, target)
            return f"Stopped app {target}."
        raise ProviderError(f"Modal cannot do '{action}'")

    async def browse(self, volume: str, path: str) -> dict[str, Any]:
        """List one directory of a Modal volume."""
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        self._configure_env()
        entries = await _run_in_modal_loop(_listdir, volume, path or "/")
        return {"path": path or "/", "entries": entries}


# --- Modal's client, driven from an async web app ---------------------------
#
# Modal's public Python API is synchronous, and its async internals only work
# inside the client's own event loop — awaiting them from ours makes RPCs hang
# with "made outside of task context". The pattern below is the one Modal's own
# CLI uses: write the work as a coroutine against the private `_`-prefixed
# classes, wrap it with `synchronizer.create_blocking`, and call that wrapper
# from a worker thread. requirements.txt pins modal because these internals are
# not a stable interface.

_wrapped: dict[str, Callable[..., Any]] = {}


async def _run_in_modal_loop(
    coroutine_fn: Callable[..., Coroutine[Any, Any, Any]], *args: Any
) -> Any:
    """Run one of the coroutines below on Modal's loop, off the web app's."""
    blocking = _wrapped.get(coroutine_fn.__name__)
    if blocking is None:
        from modal._utils.async_utils import synchronizer

        blocking = synchronizer.create_blocking(coroutine_fn)
        _wrapped[coroutine_fn.__name__] = blocking
    return await asyncio.to_thread(blocking, *args)


# App states that mean "somebody started this from a terminal and it is still
# alive". Stopping one of those is safe. Stopping a *deployed* app would take
# the lab runner offline, so the panel never offers it.
_EPHEMERAL_STATES = frozenset({"ephemeral", "detached", "detached_disconnected"})
_FINISHED_STATES = frozenset({"stopped", "stopping", "disabled"})


def _state_name(api_pb2: Any, state: int) -> str:
    """Turn APP_STATE_DEPLOYED into "deployed"."""
    for key, value in api_pb2.AppState.items():
        if value == state:
            return key.removeprefix("APP_STATE_").lower()
    return "unknown"


async def _collect(app_name: str, gpu: str) -> dict[str, Any]:
    """Read apps, containers, and volumes across every environment."""
    from modal.client import _Client
    from modal.environments import list_environments
    from modal.functions import _Function
    from modal.volume import _Volume
    from modal_proto import api_pb2

    client = await _Client.from_env()
    environments = [env.name for env in await list_environments.aio(client=client)]

    instances: list[Instance] = []
    volumes: list[Volume] = []
    apps: list[dict[str, Any]] = []
    deployed = False

    for env in environments:
        listed = await client.stub.AppList(api_pb2.AppListRequest(environment_name=env))
        for item in listed.apps:
            state = _state_name(api_pb2, item.state)
            name = item.name or item.description or item.app_id
            if name == app_name and state == "deployed":
                deployed = True
            if state in _FINISHED_STATES:
                continue
            apps.append(
                {
                    "id": item.app_id,
                    "name": name,
                    "state": state,
                    "environment": env,
                    "running_tasks": item.n_running_tasks,
                    "created_at": item.created_at or None,
                }
            )
            if state in _EPHEMERAL_STATES:
                instances.append(
                    Instance(
                        provider="modal",
                        kind="app",
                        id=item.app_id,
                        label=name,
                        status=state,
                        active=True,
                        started_at=item.created_at or None,
                        detail=(
                            f"Ephemeral app in {env}, {item.n_running_tasks} "
                            "container(s). Started by `modal run`, and it bills "
                            "until it stops."
                        ),
                        actions=[
                            {
                                "action": "stop-app",
                                "label": "Stop app",
                                "confirm": f"Stop the ephemeral app {name}?",
                            }
                        ],
                    )
                )

        running = await client.stub.TaskList(
            api_pb2.TaskListRequest(environment_name=env)
        )
        for task in running.tasks:
            ours = task.app_description == app_name
            instances.append(
                Instance(
                    provider="modal",
                    kind="container",
                    id=task.task_id,
                    label=task.app_description or task.app_id,
                    status="running" if task.started_at else "starting",
                    active=True,
                    gpu=gpu if ours else "",
                    started_at=task.started_at or task.enqueued_at or None,
                    detail=f"Container in {env}, app {task.app_id}."
                    + ("" if ours else " Not this course's app."),
                    actions=[
                        {
                            "action": "stop-container",
                            "label": "Stop container",
                            "confirm": "Stop this container? Any lab running in "
                            "it is interrupted.",
                        }
                    ],
                )
            )

        for volume in await _Volume.objects.list(environment_name=env, client=client):
            volumes.append(
                Volume(
                    provider="modal",
                    id=volume.object_id,
                    name=volume.name or volume.object_id,
                    kind="volume",
                    detail=f"Modal volume in the {env} environment.",
                    browsable=True,
                    console_url="https://modal.com/storage",
                    primary=volume.name == "learn-inference-models",
                )
            )

    functions = []
    for function_name in ("run_lab", "run_lab_cpu"):
        entry: dict[str, Any] = {"name": function_name}
        try:
            function = _Function.from_name(app_name, function_name)
            await function.hydrate(client=client)
            stats = await function.get_current_stats()
            entry.update(
                {
                    "deployed": True,
                    "backlog": stats.backlog,
                    "containers": stats.num_total_tasks,
                }
            )
        except Exception as exc:  # noqa: BLE001 - "not deployed yet" is normal
            entry.update({"deployed": False, "error": str(exc)[:200]})
        functions.append(entry)

    return {
        "environments": environments,
        "apps": apps,
        "instances": instances,
        "volumes": volumes,
        "functions": functions,
        "deployed": deployed,
    }


async def _stop_container(task_id: str) -> None:
    from modal._utils.grpc_utils import retry_transient_errors
    from modal.client import _Client
    from modal_proto import api_pb2

    # ContainerStop carries only the task id in modal 1.1.4, and the server
    # sends the container a SIGINT that Modal handles, the same as
    # `modal container stop`.
    client = await _Client.from_env()
    await retry_transient_errors(
        client.stub.ContainerStop, api_pb2.ContainerStopRequest(task_id=task_id)
    )


async def _stop_app(app_id: str) -> None:
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    await client.stub.AppStop(
        api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_WEB)
    )


async def _listdir(volume_name: str, path: str) -> list[Entry]:
    from modal.client import _Client
    from modal.volume import FileEntryType, _Volume

    client = await _Client.from_env()
    volume = _Volume.from_name(volume_name)
    await volume.hydrate(client=client)

    entries = []
    for item in await volume.listdir(path):
        entries.append(
            Entry(
                name=item.path.rstrip("/").rsplit("/", 1)[-1] or item.path,
                path=item.path,
                is_dir=item.type == FileEntryType.DIRECTORY,
                size=item.size,
                modified_at=float(item.mtime) if item.mtime else None,
            )
        )
    entries.sort(key=lambda entry: (not entry.is_dir, entry.name))
    return entries

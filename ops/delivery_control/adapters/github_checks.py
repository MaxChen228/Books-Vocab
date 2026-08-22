"""Required-check observation with exact pull-request HEAD binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from ..domain.errors import CompareAndSwapConflict
from ..domain.models import CheckStatus
from ..domain.observations import CheckSnapshot, PullRequestSnapshot
from .errors import AdapterPayloadError
from .github_client import GitHubCliClient
from .github_required_batch import batch_required_snapshots
from .timestamps import parse_optional_timestamp


def _status_from_states(states: set[str]) -> CheckStatus:
    if not states:
        return CheckStatus.ABSENT
    if states & {
        "FAILURE",
        "ERROR",
        "CANCELLED",
        "TIMED_OUT",
        "ACTION_REQUIRED",
    }:
        return CheckStatus.FAILURE
    if states <= {"SUCCESS", "SKIPPED", "NEUTRAL"}:
        return CheckStatus.SUCCESS
    return CheckStatus.PENDING


class GitHubChecks:
    def __init__(
        self,
        *,
        client: GitHubCliClient,
        get_pull_request: Callable[[int], PullRequestSnapshot],
    ) -> None:
        self.client = client
        self.get_pull_request = get_pull_request
        self._batched_required: dict[int, CheckSnapshot] = {}

    def prime_required_snapshots(self, numbers: tuple[int, ...]) -> None:
        """Prime an observation-only cache using bounded GraphQL batches."""

        self._batched_required = batch_required_snapshots(self.client, numbers)

    def _required_snapshot_live(
        self, number: int, *, before: PullRequestSnapshot
    ) -> CheckSnapshot:
        payload = self.client.load_json(
            (
                "gh",
                "pr",
                "checks",
                str(number),
                "--required",
                "--json",
                "name,state,startedAt,completedAt",
            ),
            allow_nonzero=True,
        )
        if not isinstance(payload, list):
            raise AdapterPayloadError("GitHub required checks must be a JSON list")
        names: list[str] = []
        states: set[str] = set()
        starts: list[datetime | None] = []
        completions: list[datetime | None] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise AdapterPayloadError(f"required check[{index}] is not an object")
            name = item.get("name")
            state = item.get("state")
            if type(name) is not str or type(state) is not str:
                raise AdapterPayloadError(f"required check[{index}] is malformed")
            names.append(name)
            states.add(state.upper())
            starts.append(
                parse_optional_timestamp(
                    item.get("startedAt"),
                    field=f"required check[{index}] startedAt",
                )
            )
            completions.append(
                parse_optional_timestamp(
                    item.get("completedAt"),
                    field=f"required check[{index}] completedAt",
                )
            )
        after = self.get_pull_request(number)
        if before.head_sha != after.head_sha:
            raise CompareAndSwapConflict("PR HEAD changed while reading required checks")
        return CheckSnapshot(
            status=_status_from_states(states),
            head_sha=after.head_sha,
            observed_at=datetime.now(tz=UTC),
            names=tuple(sorted(names)),
            started_at=(
                min(item for item in starts if item is not None)
                if starts and all(item is not None for item in starts)
                else None
            ),
            completed_at=(
                max(item for item in completions if item is not None)
                if completions and all(item is not None for item in completions)
                else None
            ),
        )

    def required_snapshot(self, number: int) -> CheckSnapshot:
        before = self.get_pull_request(number)
        batched = self._batched_required.pop(number, None)
        if batched is not None and batched.head_sha == before.head_sha:
            return batched
        return self._required_snapshot_live(number, before=before)

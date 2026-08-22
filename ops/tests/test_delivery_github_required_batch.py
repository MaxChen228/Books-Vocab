from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))

from delivery_control.adapters.errors import AdapterPayloadError
from delivery_control.adapters.github_checks import GitHubChecks
from delivery_control.adapters.github_client import GitHubCliClient
from delivery_control.domain.models import CheckStatus
from delivery_control.domain.observations import PullRequestSnapshot
from delivery_control.ports.process import CommandResult

HEAD = "a" * 40
BASE = "b" * 40
_DEFAULT_PAGE_INFO = object()


class StaticRunner:
    def __init__(self, responses: list[CommandResult]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: tuple[str, ...], *, cwd: Path | None = None) -> CommandResult:
        self.calls.append(argv)
        if not self.responses:
            raise AssertionError(f"unexpected command: {argv}")
        return self.responses.pop(0)


def _pr(number: int, *, head: str = HEAD) -> PullRequestSnapshot:
    return PullRequestSnapshot(
        number=number,
        url=f"https://example.test/pull/{number}",
        branch=f"feat/{number}",
        base_sha=BASE,
        head_sha=head,
        state="OPEN",
        draft=False,
        mergeable=True,
        node_id=f"PR_{number}",
        body="body",
    )


def _context_payload(
    *,
    number: int,
    head: str = HEAD,
    conclusion: str = "SUCCESS",
    required: bool = True,
    page_info: object = _DEFAULT_PAGE_INFO,
) -> dict[str, object]:
    if page_info is _DEFAULT_PAGE_INFO:
        page_info = {"hasNextPage": False}
    return {
        "data": {
            "repository": {
                f"pr_{number}": {
                    "number": number,
                    "headRefOid": head,
                    "commits": {
                        "nodes": [
                            {
                                "commit": {
                                    "statusCheckRollup": {
                                        "contexts": {
                                            "pageInfo": page_info,
                                            "nodes": [
                                                {
                                                    "__typename": "CheckRun",
                                                    "name": "required",
                                                    "status": "COMPLETED",
                                                    "conclusion": conclusion,
                                                    "startedAt": "2026-08-23T00:00:00Z",
                                                    "completedAt": "2026-08-23T00:00:01Z",
                                                    "isRequired": required,
                                                },
                                                {
                                                    "__typename": "CheckRun",
                                                    "name": "advisory",
                                                    "status": "COMPLETED",
                                                    "conclusion": "FAILURE",
                                                    "startedAt": "2026-08-23T00:00:00Z",
                                                    "completedAt": "2026-08-23T00:00:01Z",
                                                    "isRequired": False,
                                                },
                                            ]
                                        }
                                    }
                                }
                            }
                        ]
                    },
                }
            }
        }
    }


def _checks(runner: StaticRunner, number: int = 12) -> GitHubChecks:
    client = GitHubCliClient(repo=Path("/repo"), runner=runner)
    return GitHubChecks(client=client, get_pull_request=lambda _: _pr(number))


def test_batch_required_snapshot_filters_advisory_and_consumes_once() -> None:
    runner = StaticRunner(
        [
            CommandResult(
                ("gh", "repo", "view"),
                0,
                json.dumps({"nameWithOwner": "owner/repo"}),
                "",
            ),
            CommandResult(
                ("gh", "api", "graphql"),
                0,
                json.dumps(_context_payload(number=12)),
                "",
            ),
        ]
    )
    checks = _checks(runner)

    checks.prime_required_snapshots((12,))
    snapshot = checks.required_snapshot(12)

    assert snapshot.status is CheckStatus.SUCCESS
    assert snapshot.names == ("required",)
    assert snapshot.head_sha == HEAD
    assert len(runner.calls) == 2
    query = next(part for part in runner.calls[1] if part.startswith("query="))
    assert "pullRequest(number: 12)" in query


@pytest.mark.parametrize(
    ("page_info", "message"),
    [
        ({"hasNextPage": True}, "hasNextPage=true"),
        ({}, "pageInfo is malformed"),
        (None, "pageInfo is malformed"),
    ],
)
def test_batch_rejects_incomplete_required_context_connection(
    page_info: object, message: str
) -> None:
    runner = StaticRunner(
        [
            CommandResult(
                ("gh", "repo", "view"),
                0,
                json.dumps({"nameWithOwner": "owner/repo"}),
                "",
            ),
            CommandResult(
                ("gh", "api", "graphql"),
                0,
                json.dumps(_context_payload(number=12, page_info=page_info)),
                "",
            ),
        ]
    )
    checks = _checks(runner)

    with pytest.raises(AdapterPayloadError, match=message):
        checks.prime_required_snapshots((12,))


def test_batch_head_drift_falls_back_to_exact_live_check() -> None:
    runner = StaticRunner(
        [
            CommandResult(
                ("gh", "repo", "view"),
                0,
                json.dumps({"nameWithOwner": "owner/repo"}),
                "",
            ),
            CommandResult(
                ("gh", "api", "graphql"),
                0,
                json.dumps(_context_payload(number=12, head="c" * 40)),
                "",
            ),
            CommandResult(
                ("gh", "pr", "checks"),
                0,
                json.dumps(
                    [
                        {
                            "name": "required",
                            "state": "SUCCESS",
                            "startedAt": None,
                            "completedAt": None,
                        }
                    ]
                ),
                "",
            ),
        ]
    )
    checks = _checks(runner)

    checks.prime_required_snapshots((12,))
    snapshot = checks.required_snapshot(12)

    assert snapshot.status is CheckStatus.SUCCESS
    assert any(call[:3] == ("gh", "pr", "checks") for call in runner.calls)


def test_batch_rejects_malformed_repository_identity() -> None:
    runner = StaticRunner(
        [
            CommandResult(
                ("gh", "repo", "view"),
                0,
                json.dumps({"nameWithOwner": "not/a/repository/name"}),
                "",
            )
        ]
    )
    checks = _checks(runner)

    with pytest.raises(AdapterPayloadError, match="owner/name"):
        checks.prime_required_snapshots((12,))

"""Bounded GraphQL batch reader for required GitHub check observations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime

from ..domain.models import CheckStatus
from ..domain.observations import CheckSnapshot
from .errors import AdapterPayloadError
from .github_client import GitHubCliClient
from .timestamps import parse_optional_timestamp

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_BATCH_SIZE = 50


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


def _required_checks_query(numbers: tuple[int, ...]) -> str:
    fragments: list[str] = []
    for number in numbers:
        fragments.append(
            f"""
      pr_{number}: pullRequest(number: {number}) {{
        number
        headRefOid
        commits(last: 1) {{
          nodes {{
            commit {{
              statusCheckRollup {{
                contexts(first: 100) {{
                  pageInfo {{
                    hasNextPage
                  }}
                  nodes {{
                    __typename
                    ... on CheckRun {{
                      name
                      status
                      conclusion
                      startedAt
                      completedAt
                      isRequired(pullRequestNumber: {number})
                    }}
                    ... on StatusContext {{
                      context
                      state
                      isRequired(pullRequestNumber: {number})
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}"""
        )
    return (
        "query DeliveryRequiredChecks($owner: String!, $name: String!) {"
        " repository(owner: $owner, name: $name) {"
        + "".join(fragments)
        + " } }"
    )


def _repository_name(client: GitHubCliClient) -> tuple[str, str]:
    payload = client.load_json(("gh", "repo", "view", "--json", "nameWithOwner"))
    if not isinstance(payload, Mapping) or type(payload.get("nameWithOwner")) is not str:
        raise AdapterPayloadError("GitHub repository identity is malformed")
    owner, separator, name = payload["nameWithOwner"].partition("/")
    if not separator or not owner or not name or "/" in name:
        raise AdapterPayloadError("GitHub repository identity must be owner/name")
    return owner, name


def _snapshot_from_node(node: object, *, number: int) -> CheckSnapshot:
    if not isinstance(node, Mapping):
        raise AdapterPayloadError(f"required PR {number} is malformed")
    if node.get("number") != number:
        raise AdapterPayloadError(f"required PR {number} number does not match alias")
    head_sha = node.get("headRefOid")
    if type(head_sha) is not str or _SHA_RE.fullmatch(head_sha) is None:
        raise AdapterPayloadError(f"required PR {number} HEAD is malformed")
    commits = node.get("commits")
    commit_nodes = commits.get("nodes") if isinstance(commits, Mapping) else None
    if not isinstance(commit_nodes, list) or len(commit_nodes) > 1:
        raise AdapterPayloadError(f"required PR {number} commit connection is malformed")
    commit = commit_nodes[0].get("commit") if commit_nodes else None
    rollup = commit.get("statusCheckRollup") if isinstance(commit, Mapping) else None
    contexts = rollup.get("contexts") if isinstance(rollup, Mapping) else None
    if not isinstance(contexts, Mapping):
        raise AdapterPayloadError(f"required PR {number} check connection is malformed")
    page_info = contexts.get("pageInfo")
    if not isinstance(page_info, Mapping) or type(page_info.get("hasNextPage")) is not bool:
        raise AdapterPayloadError(
            f"required PR {number} check connection pageInfo is malformed"
        )
    if page_info["hasNextPage"]:
        raise AdapterPayloadError(
            f"required PR {number} check connection hasNextPage=true"
        )
    context_nodes = contexts.get("nodes")
    if not isinstance(context_nodes, list):
        raise AdapterPayloadError(f"required PR {number} check connection is malformed")

    names: list[str] = []
    states: set[str] = set()
    starts: list[datetime | None] = []
    completions: list[datetime | None] = []
    for index, item in enumerate(context_nodes):
        if not isinstance(item, Mapping):
            raise AdapterPayloadError(f"required PR {number} check[{index}] is malformed")
        typename = item.get("__typename")
        if typename == "CheckRun":
            name = item.get("name")
            required = item.get("isRequired")
            raw_status = item.get("status")
            conclusion = item.get("conclusion")
            if type(name) is not str or type(required) is not bool:
                raise AdapterPayloadError(
                    f"required PR {number} check[{index}] is malformed"
                )
            if not required:
                continue
            if type(raw_status) is not str:
                raise AdapterPayloadError(
                    f"required PR {number} check[{index}] status is malformed"
                )
            state = (
                conclusion
                if raw_status.upper() == "COMPLETED" and type(conclusion) is str
                else raw_status
            )
            starts.append(
                parse_optional_timestamp(
                    item.get("startedAt"),
                    field=f"required PR {number} check[{index}] startedAt",
                )
            )
            completions.append(
                parse_optional_timestamp(
                    item.get("completedAt"),
                    field=f"required PR {number} check[{index}] completedAt",
                )
            )
        elif typename == "StatusContext":
            name = item.get("context")
            required = item.get("isRequired")
            state = item.get("state")
            if type(name) is not str or type(required) is not bool or type(state) is not str:
                raise AdapterPayloadError(
                    f"required PR {number} status context[{index}] is malformed"
                )
            if not required:
                continue
            starts.append(None)
            completions.append(None)
        else:
            raise AdapterPayloadError(
                f"required PR {number} check[{index}] has unknown type"
            )
        names.append(name)
        states.add(state.upper())

    return CheckSnapshot(
        status=_status_from_states(states),
        head_sha=head_sha,
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


def batch_required_snapshots(
    client: GitHubCliClient, numbers: tuple[int, ...]
) -> dict[int, CheckSnapshot]:
    if type(numbers) is not tuple or any(
        type(number) is not int or number <= 0 for number in numbers
    ):
        raise AdapterPayloadError("required PR numbers must be positive integers")
    if len(set(numbers)) != len(numbers):
        raise AdapterPayloadError("required PR numbers must be unique")
    if not numbers:
        return {}
    owner, name = _repository_name(client)
    snapshots: dict[int, CheckSnapshot] = {}
    for start in range(0, len(numbers), _MAX_BATCH_SIZE):
        chunk = numbers[start : start + _MAX_BATCH_SIZE]
        payload = client.load_json(
            (
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={_required_checks_query(chunk)}",
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
            )
        )
        if not isinstance(payload, Mapping) or payload.get("errors"):
            raise AdapterPayloadError(
                "GitHub required-check GraphQL response contains errors"
            )
        data = payload.get("data")
        repository = data.get("repository") if isinstance(data, Mapping) else None
        if not isinstance(repository, Mapping):
            raise AdapterPayloadError("GitHub required-check repository is malformed")
        for number in chunk:
            snapshots[number] = _snapshot_from_node(
                repository.get(f"pr_{number}"), number=number
            )
    return snapshots

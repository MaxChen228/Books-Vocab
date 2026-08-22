"""Pure projection of Git branch refs into safe lifecycle dispositions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from ..domain.branch_lifecycle import (
    BranchAsset,
    BranchCleanupAction,
    BranchDisposition,
    BranchLifecycleInventory,
    BranchSide,
)
from ..domain.branch_refs import BranchInventory
from ..domain.observations import (
    PhysicalWorktree,
    PullRequestSnapshot,
    RegistrySnapshot,
    WorktreeSnapshot,
)

_ACTIVE_STATUSES = frozenset({"active", "published", "cleanup_pending"})
_TERMINAL_STATUSES = frozenset({"merged", "abandoned"})
_DEFAULT_PROTECTED = frozenset({"main", "prod"})


def _by_branch[T](items: tuple[T, ...], key: str) -> dict[str, tuple[T, ...]]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for item in items:
        grouped[getattr(item, key)].append(item)
    return {name: tuple(values) for name, values in grouped.items()}


def _text_paths(items: tuple[PhysicalWorktree, ...]) -> tuple[str, ...]:
    return tuple(sorted(str(item.path.resolve()) for item in items))


def _dirty_paths(
    items: tuple[PhysicalWorktree, ...],
    snapshots: Mapping[Path, WorktreeSnapshot | None],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(item.path.resolve())
            for item in items
            if (snapshot := snapshots.get(item.path.resolve())) is not None
            and not snapshot.clean
        )
    )


def _expected_heads(
    records: tuple[RegistrySnapshot, ...],
    pull_requests: tuple[PullRequestSnapshot, ...],
) -> frozenset[str]:
    values = {
        record.handed_back_sha
        for record in records
        if record.handed_back_sha is not None
    }
    values.update(item.head_sha for item in pull_requests)
    return frozenset(values)


def _unique_heads(values: tuple[str | None, ...]) -> frozenset[str]:
    return frozenset(value for value in values if value is not None)


def _live_evidence_conflicts(
    records: tuple[RegistrySnapshot, ...],
    pull_requests: tuple[PullRequestSnapshot, ...],
) -> bool:
    """Reject contradictory evidence before classifying a live lane.

    A branch ref may be equal to one of several observed heads while the
    claims themselves disagree.  Membership in the union is therefore not
    enough: it could hide a registry/PR split and incorrectly preserve or
    clean the wrong asset.
    """

    live_records = tuple(item for item in records if item.status in _ACTIVE_STATUSES)
    open_prs = tuple(item for item in pull_requests if item.state == "OPEN")
    record_heads = _unique_heads(tuple(item.handed_back_sha for item in live_records))
    pr_heads = _unique_heads(tuple(item.head_sha for item in open_prs))
    if len(record_heads) > 1 or len(pr_heads) > 1:
        return True
    return bool(record_heads and pr_heads and record_heads != pr_heads)


def _exact_merged_evidence(
    records: tuple[RegistrySnapshot, ...],
    pull_requests: tuple[PullRequestSnapshot, ...],
) -> bool:
    """Require one complete merged proof before exposing cleanup-ready."""

    merged_records = tuple(item for item in records if item.status == "merged")
    merged_prs = tuple(item for item in pull_requests if item.state == "MERGED")
    if len(records) != 1 or len(pull_requests) != 1:
        return False
    if len(merged_records) != 1 or len(merged_prs) != 1:
        return False
    record_head = merged_records[0].handed_back_sha
    return record_head is not None and record_head == merged_prs[0].head_sha


def _asset(
    *,
    branch: str,
    side: BranchSide,
    sha: str,
    disposition: BranchDisposition,
    action: BranchCleanupAction,
    reason: str,
    records: tuple[RegistrySnapshot, ...],
    pull_requests: tuple[PullRequestSnapshot, ...],
    physical: tuple[PhysicalWorktree, ...],
    snapshots: Mapping[Path, WorktreeSnapshot | None],
) -> BranchAsset:
    return BranchAsset(
        branch=branch,
        side=side,
        sha=sha,
        disposition=disposition,
        cleanup_action=action,
        reason=reason,
        protected=disposition is BranchDisposition.PROTECTED,
        pull_request_numbers=tuple(sorted(item.number for item in pull_requests)),
        registry_statuses=tuple(sorted({item.status for item in records})),
        physical_worktree_paths=_text_paths(physical),
        dirty_worktree_paths=_dirty_paths(physical, snapshots),
    )


def _project_asset(
    *,
    branch: str,
    side: BranchSide,
    sha: str,
    records: tuple[RegistrySnapshot, ...],
    pull_requests: tuple[PullRequestSnapshot, ...],
    physical: tuple[PhysicalWorktree, ...],
    snapshots: Mapping[Path, WorktreeSnapshot | None],
    protected_branches: frozenset[str],
) -> BranchAsset:
    if branch in protected_branches:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.PROTECTED,
            action=BranchCleanupAction.PRESERVE_PROTECTED,
            reason="protected branch or explicitly retained backup",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )

    live_records = tuple(item for item in records if item.status in _ACTIVE_STATUSES)
    expected_heads = _expected_heads(records, pull_requests)
    if len(records) > 1 and live_records:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.UNKNOWN,
            action=BranchCleanupAction.INSPECT_UNKNOWN,
            reason="multiple live registry claims share one branch",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if _live_evidence_conflicts(records, pull_requests):
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.UNKNOWN,
            action=BranchCleanupAction.INSPECT_UNKNOWN,
            reason="live registry and PR evidence disagree on branch HEAD",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if expected_heads and sha not in expected_heads:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.REMOTE_DRIFT,
            action=BranchCleanupAction.PRESERVE_REMOTE_DRIFT,
            reason="branch SHA differs from every typed handback or PR head",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if live_records:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.ACTIVE_OR_PUBLISHED_LANE,
            action=BranchCleanupAction.FOLLOW_OWNER_LANE,
            reason="branch is owned by an active or published delivery lane",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    open_prs = tuple(item for item in pull_requests if item.state == "OPEN")
    if len(open_prs) > 1:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.UNKNOWN,
            action=BranchCleanupAction.INSPECT_UNKNOWN,
            reason="multiple open PRs share one branch",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if open_prs:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.OPEN_PR_DURABLE,
            action=BranchCleanupAction.PRESERVE_DURABLE_PR,
            reason="unique open PR makes the remote branch durable queue state",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    merged_prs = tuple(item for item in pull_requests if item.state == "MERGED")
    merged_records = tuple(item for item in records if item.status == "merged")
    dirty = bool(_dirty_paths(physical, snapshots))
    if (
        merged_prs
        and merged_records
        and not _exact_merged_evidence(records, pull_requests)
    ):
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.UNKNOWN,
            action=BranchCleanupAction.INSPECT_UNKNOWN,
            reason="merged registry and PR evidence is not one exact terminal proof",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if _exact_merged_evidence(records, pull_requests) and not dirty:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.MERGED_CLEANUP_READY,
            action=BranchCleanupAction.CLEANUP_MERGED,
            reason="merged PR and terminal registry proof agree on branch SHA",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if any(item.status == "abandoned" and item.handed_back_sha for item in records):
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.ABANDONED_WITH_HANDBACK,
            action=BranchCleanupAction.RECOVER_OWNER_OR_REQUIRE_DISCARD_PROOF,
            reason="abandoned record retains product handback without PR proof",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if any(item.state == "CLOSED" for item in pull_requests):
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.CLOSED_DISPOSITION_REQUIRED,
            action=BranchCleanupAction.RECONCILE_CLOSED_PR,
            reason="closed unmerged PR requires explicit terminal disposition",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    if side is BranchSide.LOCAL:
        return _asset(
            branch=branch,
            side=side,
            sha=sha,
            disposition=BranchDisposition.ORPHAN_LOCAL_RECONCILE,
            action=BranchCleanupAction.RECONCILE_LOCAL_ORPHAN,
            reason="local branch has no active lane, PR, or terminal proof",
            records=records,
            pull_requests=pull_requests,
            physical=physical,
            snapshots=snapshots,
        )
    return _asset(
        branch=branch,
        side=side,
        sha=sha,
        disposition=BranchDisposition.ORPHAN_REMOTE_RECONCILE,
        action=BranchCleanupAction.RECONCILE_REMOTE_ORPHAN,
        reason="remote branch has no active lane, PR, or terminal proof",
        records=records,
        pull_requests=pull_requests,
        physical=physical,
        snapshots=snapshots,
    )


def project_branch_lifecycle(
    *,
    branch_inventory: BranchInventory,
    records: tuple[RegistrySnapshot, ...] = (),
    pull_requests: tuple[PullRequestSnapshot, ...] = (),
    physical: tuple[PhysicalWorktree, ...] = (),
    snapshots: Mapping[Path, WorktreeSnapshot | None] | None = None,
    protected_branches: frozenset[str] = _DEFAULT_PROTECTED,
) -> BranchLifecycleInventory:
    """Partition every observed ref exactly once without performing mutation."""

    snapshots = snapshots or {}
    records_by_branch = _by_branch(records, "branch")
    prs_by_branch = _by_branch(pull_requests, "branch")
    physical_by_branch: dict[str, tuple[PhysicalWorktree, ...]] = defaultdict(tuple)
    grouped_physical: dict[str, list[PhysicalWorktree]] = defaultdict(list)
    for item in physical:
        if item.branch is not None:
            grouped_physical[item.branch].append(item)
    physical_by_branch = {
        branch: tuple(items) for branch, items in grouped_physical.items()
    }

    assets: list[BranchAsset] = []
    for side, refs in (
        (BranchSide.LOCAL, branch_inventory.local),
        (BranchSide.REMOTE, branch_inventory.remote),
    ):
        for branch, sha in refs:
            assets.append(
                _project_asset(
                    branch=branch,
                    side=side,
                    sha=sha,
                    records=records_by_branch.get(branch, ()),
                    pull_requests=prs_by_branch.get(branch, ()),
                    physical=physical_by_branch.get(branch, ()),
                    snapshots=snapshots,
                    protected_branches=protected_branches,
                )
            )
    return BranchLifecycleInventory(assets=tuple(assets))


__all__ = ["project_branch_lifecycle"]

#!/usr/bin/env bash
# pin_actions.sh -- rewrite every `uses: owner/repo@tag` in .github/workflows
# to the immutable commit SHA that tag points at today.
#
# WHY
#   `uses: actions/checkout@v4` does not name a version.  It names a TAG, and a
#   tag is a mutable pointer the action's owner can move at any time.  Whoever
#   controls that repository -- or anyone who compromises it -- can change what
#   runs inside this build, with write access to the workspace and whatever
#   scopes GITHUB_TOKEN carries.  This is not hypothetical; it is how the
#   tj-actions/changed-files compromise reached tens of thousands of
#   repositories in March 2025.
#
#   A 40-character commit SHA cannot be moved.  Pinning costs one command and a
#   comment saying which version the SHA was, so upgrades stay legible.
#
# WHY THIS IS A SCRIPT AND NOT ALREADY DONE
#   The SHAs have to be resolved against GitHub by someone who can reach it and
#   who can check what they are pinning.  Writing SHAs into a workflow without
#   verifying them is the same trust failure in the other direction.
#
# USAGE
#   gh auth login                      # once
#   bash tools/pin_actions.sh          # rewrite in place
#   bash tools/pin_actions.sh --check  # exit 1 if anything is unpinned
#
#   Then flip `continue-on-error: true` to false on the supply-chain job in
#   .github/workflows/ci.yml, so it stays pinned.
#
# UPGRADING LATER
#   Re-run this script; it re-resolves each tag from the trailing comment it
#   wrote.  Or use dependabot, which understands SHA pins and opens PRs that
#   move them with the version comment updated.
set -euo pipefail

cd "$(dirname "$0")/.."
check_only=0
[ "${1:-}" = "--check" ] && check_only=1

command -v gh >/dev/null || { echo "need the gh CLI: https://cli.github.com"; exit 2; }
if [ "$check_only" = 0 ]; then
    gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated; run: gh auth login"; exit 2; }
fi

unpinned=0
for f in .github/workflows/*.yml .github/workflows/*.yaml; do
    [ -e "$f" ] || continue
    while IFS= read -r spec; do
        repo="${spec%@*}"
        ref="${spec##*@}"
        # Already a SHA?  Leave it alone.
        if printf '%s' "$ref" | grep -Eq '^[0-9a-f]{40}$'; then continue; fi
        unpinned=1
        if [ "$check_only" = 1 ]; then
            echo "UNPINNED  $f: $spec"
            continue
        fi
        # An action may live in a subdirectory of its repository --
        # github/codeql-action/init lives in github/codeql-action.
        owner_repo="$(printf '%s' "$repo" | cut -d/ -f1,2)"
        sha="$(gh api "repos/${owner_repo}/commits/${ref}" --jq .sha 2>/dev/null || true)"
        if ! printf '%s' "$sha" | grep -Eq '^[0-9a-f]{40}$'; then
            echo "could not resolve ${owner_repo}@${ref} -- leaving it alone" >&2
            continue
        fi
        echo "  ${spec}  ->  ${sha}  # ${ref}"
        # The trailing comment is what makes the pin readable and re-resolvable.
        perl -pi -e "s{\Q${spec}\E(?!\s*#)}{${repo}\@${sha} # ${ref}}g" "$f"
    done < <(grep -hoE 'uses:[[:space:]]*[A-Za-z0-9._/-]+@[A-Za-z0-9._-]+' "$f" \
             | sed 's/uses:[[:space:]]*//' | sort -u)
done

if [ "$check_only" = 1 ]; then
    if [ "$unpinned" = 0 ]; then echo "every action is pinned to a commit SHA"; exit 0; fi
    exit 1
fi
echo
echo "done -- review the diff before committing, and check each SHA is the one you meant:"
echo "    git diff .github/workflows"

#!/usr/bin/env bash
# ===========================================================================
#  push.sh — put this repository on GitHub.
#
#  Run it from inside the igaos/ folder:
#
#      cd igaos
#      bash push.sh
#
#  It does not want a token typed at it and it does not store one. Either the
#  GitHub CLI is installed and already signed in, in which case it uses that,
#  or it hands the push to git and git asks you in its own way (Keychain on a
#  Mac, or a browser sign-in). Nothing here reads, prints, or writes a
#  credential.
#
#  DEFAULT IS PRIVATE. Read the note at the bottom before you make it public:
#  the licence in this repository reserves all rights not granted to MRPL and
#  the Ministry, so a public repo is readable by anyone but reusable by no one.
#  That is a deliberate position, not an accident, but it should be a decision.
# ===========================================================================
set -euo pipefail

REPO_NAME="${REPO_NAME:-igaos}"
VISIBILITY="${VISIBILITY:-private}"      # private | public

cd "$(dirname "$0")"

# --- sanity: are we actually in the repository? ----------------------------
if [ ! -d .git ]; then
    echo "error: no .git here. Run this from inside the igaos/ folder." >&2
    exit 1
fi

echo "repository : $(pwd)"
echo "commit     : $(git log --oneline -1)"
echo "branch     : $(git rev-parse --abbrev-ref HEAD)"
dirty=$(git status --porcelain | wc -l | tr -d ' ')
echo "uncommitted: ${dirty}"
if [ "${dirty}" != "0" ]; then
    echo
    echo "There are uncommitted changes. Commit them first, or they will not go up:"
    echo "    git add -A && git commit -m 'your message'"
    exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo
    echo "This repository already has an origin:"
    echo "    $(git remote get-url origin)"
    echo "Pushing to it."
    git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
    echo "done."
    exit 0
fi

echo
echo "visibility : ${VISIBILITY}   (override with  VISIBILITY=public bash push.sh)"
echo "name       : ${REPO_NAME}    (override with  REPO_NAME=something bash push.sh)"
echo

# --- path A: the GitHub CLI ------------------------------------------------
if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        echo "Using the GitHub CLI (already signed in)."
        gh repo create "${REPO_NAME}" \
            --"${VISIBILITY}" \
            --source=. \
            --remote=origin \
            --push \
            --description "Indigenous GPU-accelerated optimization solver — LP, QP, MILP and MIQP from mathematical foundation. SIH 2026, PS 26119 (MRPL)."
        echo
        echo "done: $(gh repo view --json url -q .url 2>/dev/null || echo "check github.com")"
        exit 0
    else
        echo "The GitHub CLI is installed but not signed in. Run:"
        echo "    gh auth login"
        echo "then run this script again. (It opens a browser; no token is typed here.)"
        exit 1
    fi
fi

# --- path B: no CLI, so create the empty repo in a browser -----------------
cat <<'EOF'
The GitHub CLI is not installed. Two options.

  Option 1 — install it, then re-run this script:
      macOS:  brew install gh && gh auth login
      Linux:  see https://github.com/cli/cli#installation

  Option 2 — do it by hand, which is two steps:

      1. Open https://github.com/new
         Name it        : igaos
         Visibility     : Private   (see the note at the end of this script)
         Do NOT tick "Add a README", ".gitignore" or "licence" — this
         repository already has all three, and an initial commit on the
         GitHub side would collide with this one.

      2. Copy the URL it gives you and run, from this folder:

             git remote add origin https://github.com/<you>/igaos.git
             git push -u origin main

         Git will ask for credentials in its own way. On a Mac that is
         usually the Keychain or a browser window.

EOF
exit 1

# ===========================================================================
#  ON MAKING IT PUBLIC
#
#  LICENSE in this repository retains copyright with the team, grants MRPL,
#  ONGC, the Ministry of Education and AICTE a perpetual royalty-free licence,
#  and reserves everything else. That is what the Ministry's deployment
#  guidelines contemplate: the IP stays with the students so a company can be
#  built on it.
#
#  A public repository under that licence is readable by anyone and reusable
#  by no one. That is legitimate — it is how you show a jury the work is real
#  without giving it away. But if you would rather the code be freely reusable,
#  change the licence BEFORE publishing, not after: a file that was public
#  under one licence has already been copied under it.
#
#  Private is the safe default and costs nothing — you can add jury members as
#  collaborators, and flip to public in one click when you have decided.
# ===========================================================================

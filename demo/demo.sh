#!/usr/bin/env bash
# IGAOS — presentation demo driver.
#
# Runs the six demo beats in order, pausing between them so you can talk.
# Nothing here is staged: every command is the real binary on a real model, and
# the script fails loudly rather than pretending a step worked.
#
#   ./demo/demo.sh            run all six beats, pausing between each
#   ./demo/demo.sh 2          run only beat 2
#   ./demo/demo.sh --check    run everything with no pauses, verify, print PASS/FAIL
#
# Beat 5 (Solver Studio) is a browser demo; this script only tells you how to
# start it, because a server needs a window you drive by hand.

set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${IGAOS_BIN:-./build/igaos}"
CHECK=0
BEAT="${1:-all}"
[[ "$BEAT" == "--check" ]] && { CHECK=1; BEAT=all; }

bold=$'\033[1m'; dim=$'\033[2m'; grn=$'\033[32m'; red=$'\033[31m'; rst=$'\033[0m'
fails=0

hdr() { printf '\n%s================================================================%s\n' "$bold" "$rst"
        printf '%s  BEAT %s — %s%s\n' "$bold" "$1" "$2" "$rst"
        printf '%s================================================================%s\n\n' "$bold" "$rst"; }
cmd() { printf '%s$ %s%s\n\n' "$dim" "$*" "$rst"; }
pause() { (( CHECK )) && return 0; printf '\n%s[enter to continue]%s ' "$dim" "$rst"; read -r _; }
ok()   { printf '%s  PASS%s  %s\n' "$grn" "$rst" "$1"; }
bad()  { printf '%s  FAIL%s  %s\n' "$red" "$rst" "$1"; fails=$((fails+1)); }
want() { # want <needle> <text> <label>
  if grep -qF -- "$1" <<<"$2"; then ok "$3"; else bad "$3 (expected to find: $1)"; fi; }

if [[ ! -x "$BIN" ]]; then
  printf '%serror:%s no solver binary at %s\n' "$red" "$rst" "$BIN" >&2
  printf 'build it first:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8\n' >&2
  exit 1
fi

beat1() {
  hdr 1 "It solves a real refinery model"
  cmd "$BIN -m blend_m"
  out=$("$BIN" -m blend_m 2>&1); echo "$out"
  want "status         optimal"  "$out" "blend_m solves to optimal"
  want "-2135858.97349"          "$out" "objective is the expected value"
}

beat2() {
  hdr 2 "Four algorithms, one answer   <-- the centrepiece"
  cmd "$BIN -m blend_s --compare paths"
  out=$("$BIN" -m blend_s --compare paths 2>&1); echo "$out"
  n=$(grep -c -- "-744955.877255" <<<"$out")
  if [[ "$n" -ge 4 ]]; then ok "all four algorithms returned the same 12 digits"
  else bad "expected 4 identical objectives, saw $n"; fi
}

beat3() {
  hdr 3 "Infeasible — and which constraints are fighting"
  cmd "$BIN benchmarks/netlib_infeasible/galenet.mps --iis"
  out=$("$BIN" benchmarks/netlib_infeasible/galenet.mps --iis 2>&1); echo "$out"
  want "irreducible" "$out" "IIS is proven irreducible"
  want "5 members"   "$out" "galenet reduces to 5 members"
}

beat4() {
  hdr 4 "What a constraint is worth"
  cmd "$BIN -m blend_s --sensitivity"
  out=$("$BIN" -m blend_s --sensitivity 2>&1); echo "$out" | head -40
  want "SHADOW PRICES" "$out" "shadow prices reported"
  want "REDUCED COSTS" "$out" "reduced costs reported"
}

beat5() {
  hdr 5 "Solver Studio — live MILP in the browser"
  printf 'Start the console in a second terminal:\n\n'
  cmd "python3 demo/server.py"
  printf '  then open  http://127.0.0.1:8420  and run "Unit commitment (uc_l)".\n\n'
  printf '  Expect: optimal 298617.164379, bound 298617.164376,\n'
  printf '          4,601 nodes, 91 cuts over 12 rounds, about 16 s.\n\n'
  printf '  %sIt takes ~16 s. Start it, cut to beat 6, cut back.%s\n' "$bold" "$rst"
  if (( CHECK )); then
    if curl -s --max-time 4 localhost:8420/api/env | grep -q '"version"'; then
      ok "Studio is already running and answering"
    else
      printf '%s  SKIP%s  Studio not running (start it by hand for the demo)\n' "$dim" "$rst"
    fi
  fi
}

beat6() {
  hdr 6 "The tests"
  cmd "cd build && ctest"
  out=$(cd build && ctest 2>&1); echo "$out" | tail -6
  want "100% tests passed" "$out" "all six suites pass"
}

run() { case "$1" in
  1) beat1;; 2) beat2;; 3) beat3;; 4) beat4;; 5) beat5;; 6) beat6;;
  *) printf 'unknown beat: %s (use 1-6)\n' "$1"; exit 2;; esac; }

if [[ "$BEAT" == all ]]; then
  for b in 1 2 3 4 5 6; do run "$b"; [[ "$b" != 6 ]] && pause; done
else
  run "$BEAT"
fi

if (( CHECK )); then
  printf '\n%s================================================================%s\n' "$bold" "$rst"
  if (( fails == 0 )); then printf '%s  ALL DEMO BEATS VERIFIED%s\n' "$grn" "$rst"
  else printf '%s  %d CHECK(S) FAILED — do not record until these pass%s\n' "$red" "$fails" "$rst"; fi
  printf '%s================================================================%s\n' "$bold" "$rst"
fi
exit $(( fails > 0 ))

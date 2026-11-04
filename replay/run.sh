#!/bin/bash
# Runs each replay task against the configured LLM and prints one line per run.
# usage: replay/run.sh BINARY OUTDIR [RUNS] [TASK...]
# A run is a fresh copy of the fixture, one `--cli -p` prompt, then the task's check.
set -u
bin=$1 out=$2 runs=${3:-3}; shift; shift; [ $# -gt 0 ] && shift
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
tasks=("$@"); [ ${#tasks[@]} -gt 0 ] || tasks=($(ls "$here/tasks"))
printf 'task\trun\tpass\tsecs\tcalls\tcompletion\tprompt_max\tevaluated\tcached\tedits_missed\tfailed_flag\tchanged\n' | tee "$out/results.tsv"
for t in "${tasks[@]}"; do
  for r in $(seq 1 "$runs"); do
    w=$(mktemp -d) && cp -r "$here/testdata/fixture/." "$w/" && (cd "$w" && git init -q && git add -A && git -c user.email=r@r -c user.name=r commit -qm base)
    log="$out/$t.$r.out"; start=$(date +%s)
    timeout 420 "$bin" --cli --cwd "$w" -p "$(cat "$here/tasks/$t/prompt")" </dev/null >"$log" 2>&1
    secs=$(( $(date +%s) - start ))
    (cd "$w" && REPLAY_OUT="$log" timeout 120 bash "$here/tasks/$t/check.sh") >/dev/null 2>&1 && pass=1 || pass=0
    s=$(ls "$w"/.codehalter/session_*.log 2>/dev/null | head -1); cp "$s" "$out/$t.$r.log" 2>/dev/null
    toml=$(ls "$w"/.codehalter/session_*.toml 2>/dev/null | head -1)
    # One RESPONSE line per model call; the prefill warm-ups ask for one token.
    read -r calls comp pmax evald cach < <(grep -h '^tokens: prompt=' "$s" 2>/dev/null | grep -v 'completion=1 finish=length' | sed -E 's/.*prompt=([0-9]+) completion=([0-9]+).*cached=([0-9]+) evaluated=([0-9]+).*/\1 \2 \3 \4/' | awk '{n++;c+=$2;if($1>m)m=$1;e+=$4;k+=$3}END{print n+0,c+0,m+0,e+0,k+0}')
    miss=$(grep -c '^ *output = "error: \(old_text not found\|old_text.*not unique\)' "$toml" 2>/dev/null); fl=$(grep -c '^ *input = .*failed.*: *true' "$toml" 2>/dev/null)
    ch=$(cd "$w" && git status --porcelain -- . ':!.codehalter' | wc -l)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$r" "$pass" "$secs" "$calls" "$comp" "$pmax" "$evald" "$cach" "$miss" "$fl" "$ch" | tee -a "$out/results.tsv"
    rm -rf "$w"
  done
done
awk -F'\t' 'NR>1{p[$1]+=$3;n[$1]++;s[$1]+=$4;c[$1]+=$6;e[$1]+=$8;P+=$3;N++;S+=$4;C+=$6;E+=$8}END{for(t in n)printf "%s pass %d/%d avg %ds gen %d eval %d\n",t,p[t],n[t],s[t]/n[t],c[t]/n[t],e[t]/n[t]|"sort";close("sort");printf "ALL pass %d/%d avg %ds gen %d eval %d\n",P,N,S/N,C/N,E/N}' "$out/results.tsv" | tee "$out/summary.txt"

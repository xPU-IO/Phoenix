#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# run_matrix.sh — Phoenix test matrix: module load modes x libphoenix modes x
# I/O APIs x raw ioctl contract, with a kernel-log/taint check after every case.
#
# The matrix itself lives in matrix.conf (format documented there).
#
# Usage: test/kernel_module_matrix/run_matrix.sh [options]
#   --ko PATH          phoenixfs.ko to insmod (default: <build>/module/phoenixfs.ko).
#                      With module.sig_enforce=1 this must be a signed copy.
#   --build DIR        build tree with bin/ (default: <repo>/build)
#   --gpu N            primary CUDA GPU (default 0; test_stream always uses 0)
#   --gpu2 N           second GPU for multi-device cases (default 1)
#   --file PATH        data file prefix on an O_DIRECT fs (default /mnt/nvme0/phxfs_test.bin)
#   --matrix FILE      matrix file (default: matrix.conf next to this script)
#   --only REGEX       run only case IDs matching REGEX (grep -E)
#   --skip REGEX       skip case IDs matching REGEX
#   --out DIR          results directory (default /tmp/phx_matrix_<timestamp>)
#   --list             print the selected cases and exit
#   --dry-run          print what would run (including module reloads), run nothing
#   --stop-on-fail     stop at the first failing case
#   --keep-loaded      leave phoenixfs loaded at the end (default: rmmod)
#   -h | --help
#
# Examples:
#   test/kernel_module_matrix/run_matrix.sh --ko /tmp/phoenixfs.ko
#   test/kernel_module_matrix/run_matrix.sh --ko /tmp/phoenixfs.ko --only '^full\.ioctl'
#   test/kernel_module_matrix/run_matrix.sh --only '^(cpu|host|auto\.nomodule)\.'   # no module needed
#
# Exit status: 0 if every selected case passed, 1 otherwise, 2 on setup error.

set -u
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD="$REPO/build"
KO=""
GPU=0
GPU2=1
FILE=/mnt/nvme0/phxfs_test.bin
MATRIX="$SCRIPT_DIR/matrix.conf"
ONLY=""
SKIPRE=""
OUT=""
LIST=0
DRY=0
STOP=0
KEEP=0

die() { echo "run_matrix: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ko) KO=$2; shift 2 ;;
        --build) BUILD=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --gpu2) GPU2=$2; shift 2 ;;
        --file) FILE=$2; shift 2 ;;
        --matrix) MATRIX=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        --skip) SKIPRE=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --list) LIST=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --stop-on-fail) STOP=1; shift ;;
        --keep-loaded) KEEP=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

BUILD=$(cd "$BUILD" 2>/dev/null && pwd) || die "build dir not found"
BIN="$BUILD/bin"
[ -n "$KO" ] || KO="$BUILD/module/phoenixfs.ko"
[ -f "$MATRIX" ] || die "matrix file not found: $MATRIX"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo -n"; fi

if [ -t 1 ]; then
    C_RED=$'\e[31m'; C_GRN=$'\e[32m'; C_YEL=$'\e[33m'; C_CYN=$'\e[36m'; C_OFF=$'\e[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_OFF=""
fi
info() { echo "${C_CYN}>>${C_OFF} $*"; }
warn() { echo "${C_YEL}!!${C_OFF} $*"; }

# ------------------------------------------------------------------ matrix

IDS=(); KMODS=(); ENVS=(); TMOS=(); EXPS=(); CMDS=()
load_matrix() {
    local line id kmod env tmo exp cmd seen=" "
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%%$'\r'}
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        read -r id kmod env tmo exp cmd <<<"$line"
        [ -n "${cmd:-}" ] || die "malformed matrix line: $line"
        case "$kmod" in full|staging|none|keep) ;; *) die "$id: bad KMOD '$kmod'" ;; esac
        case "$exp" in pass|strict|skip|fail|xfail) ;; *) die "$id: bad EXPECT '$exp'" ;; esac
        [[ "$tmo" =~ ^[0-9]+$ ]] || die "$id: bad TIMEOUT '$tmo'"
        [[ "$seen" == *" $id "* ]] && die "duplicate case id: $id"
        seen+="$id "
        if [ -n "$ONLY" ] && ! grep -Eq -- "$ONLY" <<<"$id"; then continue; fi
        if [ -n "$SKIPRE" ] && grep -Eq -- "$SKIPRE" <<<"$id"; then continue; fi
        IDS+=("$id"); KMODS+=("$kmod"); ENVS+=("$env"); TMOS+=("$tmo"); EXPS+=("$exp"); CMDS+=("$cmd")
    done <"$MATRIX"
}

expand() {
    local s=$1
    s=${s//\{BIN\}/$BIN}; s=${s//\{BUILD\}/$BUILD}; s=${s//\{REPO\}/$REPO}
    s=${s//\{GPU2\}/$GPU2}; s=${s//\{GPU\}/$GPU}; s=${s//\{FILE\}/$FILE}
    printf '%s' "$s"
}

load_matrix
[ ${#IDS[@]} -gt 0 ] || die "no case selected"

if [ $LIST -eq 1 ]; then
    printf '%-28s %-8s %-40s %-6s %s\n' ID KMOD ENV EXPECT COMMAND
    for i in "${!IDS[@]}"; do
        printf '%-28s %-8s %-40s %-6s %s\n' "${IDS[$i]}" "${KMODS[$i]}" "${ENVS[$i]}" \
            "${EXPS[$i]}" "$(expand "${CMDS[$i]}")"
    done
    exit 0
fi

# ------------------------------------------------------------------ module

mod_loaded() { grep -q '^phoenixfs ' /proc/modules; }
mod_mode() { cat /sys/module/phoenixfs/parameters/phxfs_map_mode 2>/dev/null || echo -; }
mode_num() { case "$1" in full) echo 0 ;; staging) echo 1 ;; *) echo -1 ;; esac; }
mode_name() { case "$1" in 0) echo full ;; 1) echo staging ;; *) echo none ;; esac; }
cur_mode() { if mod_loaded; then mode_name "$(mod_mode)"; else echo none; fi; }
refcnt() { cat /sys/module/phoenixfs/refcnt 2>/dev/null || echo -; }

# Same snapshot the module Makefile's refresh-pat-snapshot target takes.
refresh_pat() {
    local src=/sys/kernel/debug/x86/pat_memtype_list dst=/run/phxfs/pat_memtype_list
    $SUDO sh -c "install -d -m 0700 /run/phxfs && rm -f $dst &&
                 { [ -r $src ] && cat $src > $dst.tmp && [ -s $dst.tmp ] &&
                   chmod 0400 $dst.tmp && mv -f $dst.tmp $dst; } ||
                 { rm -f $dst.tmp; echo 'warning: PAT snapshot unavailable' >&2; }"
}

do_insmod() {   # mode extra-params...
    local m; m=$(mode_num "$1"); shift
    refresh_pat
    echo "\$ insmod $KO phxfs_map_mode=$m $*"
    $SUDO insmod "$KO" phxfs_map_mode="$m" "$@" 2>&1
}

do_rmmod() {
    mod_loaded || return 0
    echo "\$ rmmod phoenixfs (refcnt=$(refcnt))"
    timeout 120 $SUDO rmmod phoenixfs 2>&1
}

# Normalise "00000000:06:00.0" / "0000:06:00.0" to "0000:06:00.0".
norm_bdf() { local b=${1,,}; local dom=${b%%:*} rest=${b#*:}; printf '%04x:%s' "$((16#$dom))" "$rest"; }

check_load() {   # expected mode name
    local want=$1 bad=0 n=0 d bdf m p
    mod_loaded || { echo "phoenixfs not loaded"; return 1; }
    [ "$(mod_mode)" = "$(mode_num "$want")" ] || { echo "phxfs_map_mode=$(mod_mode), want $want"; bad=1; }
    local gpus; gpus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null |
                       while read -r b; do norm_bdf "$b"; echo; done)
    [ -c /dev/phxfs_dev ] || { echo "missing char device /dev/phxfs_dev"; bad=1; }
    for d in /dev/phxfs_dev[0-9]*; do
        [ ! -c "$d" ] || { echo "unexpected legacy char device $d"; bad=1; }
    done
    for d in /sys/class/phxfs-generic/phxfs_dev[0-9]*; do
        [ -d "$d" ] || continue
        n=$((n + 1))
        bdf=$(cat "$d/pci_bdf" 2>/dev/null); m=$(cat "$d/map_mode" 2>/dev/null)
        p=$(cat "$d/page_size" 2>/dev/null)
        echo "  ${d##*/}: bdf=$bdf map_mode=$m page_size=$p"
        [ "$m" = "$(mode_num "$want")" ] || { echo "  map_mode mismatch"; bad=1; }
        [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -ge 4096 ] && [ $((p & (p - 1))) -eq 0 ] ||
            { echo "  bad page_size"; bad=1; }
        if [ -n "$gpus" ] && ! grep -qix -- "$bdf" <<<"$gpus"; then
            echo "  $bdf is not an NVIDIA GPU"; bad=1
        fi
    done
    [ $n -gt 0 ] || { echo "no phxfs devices"; bad=1; }
    local want_n; want_n=$(grep -c . <<<"$gpus")
    if [ "$(cat /sys/module/phoenixfs/parameters/phxfs_numa_node 2>/dev/null)" = "-1" ] &&
       [ "$want_n" -gt 0 ] && [ "$n" -ne "$want_n" ]; then
        echo "  $n phxfs devices for $want_n GPUs"; bad=1
    fi
    echo "  node=/dev/phxfs_dev gpu_metadata=$n refcnt=$(refcnt)"
    return $bad
}

wait_refcnt_zero() {
    local i
    for i in $(seq 1 "${1:-20}"); do
        [ "$(refcnt)" = "0" ] && return 0
        sleep 0.5
    done
    echo "refcnt stuck at $(refcnt)"
    return 1
}

# ------------------------------------------------------------------ kernel check

BAD_RE='BUG:|WARNING:|Oops|general protection|Kernel panic|KASAN|UBSAN|refcount_t:|list_(add|del) corruption|soft lockup|hard LOCKUP|hung_task|blocked for more than|unbalanced put|put of unknown span|bad page state|Bad page'
TAINT_MASK=$(( (1 << 4) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 14) ))   # M B D W L
klines() { $SUDO dmesg 2>/dev/null | wc -l; }
taint() { cat /proc/sys/kernel/tainted; }

# ------------------------------------------------------------------ builtins

# GPU <ordinal>'s sysfs BDF (0000:06:00.0 form), via nvidia-smi.
gpu_bdf() {
    local b
    b=$(nvidia-smi -i "$GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null)
    [ -n "$b" ] || return 1
    norm_bdf "$b"
}

# Largest BAR of <bdf>: prints "start size" (hex, no 0x).
gpu_bar_range() {
    awk '$1!="0x0000000000000000" {s=strtonum($1); e=strtonum($2);
         if (e>s && e-s>max) {max=e-s; st=s}} END {if (max>0) printf "%x %x", st, max}' \
        "/sys/bus/pci/devices/$1/resource" 2>/dev/null
}

# True if the PAT snapshot has a non-write-back entry overlapping [start,+size).
# (Refreshes the snapshot first; takes hex start/size without 0x.)
pat_snapshot_has_wc() {
    refresh_pat
    awk -v bstr="0x$1" -v szstr="0x$2" '
        BEGIN { bs = strtonum(bstr); be = bs + strtonum(szstr) }
        /write-back/ { next }
        match($0, /0x[0-9a-fA-F]+-0x[0-9a-fA-F]+/) {
            split(substr($0, RSTART, RLENGTH), r, "-")
            cs = strtonum(r[1]); ce = strtonum(r[2])
            if (cs < be && ce > bs) { found = 1 }
        }
        END { exit found ? 0 : 1 }' /run/phxfs/pat_memtype_list 2>/dev/null
}

# Start bar_holder in the background (mode args...); sets H_PID/H_LOG.
wc_holder_start() {   # bdf mode [offMiB lenMiB] seconds
    local bdf=$1; shift
    H_LOG="$CASE_DIR/wc_holder.log"
    $SUDO "$BIN/bar_holder" "$bdf" "$@" >"$H_LOG" 2>&1 &
    H_PID=$!
    local i
    for i in $(seq 1 40); do
        grep -q HOLDING "$H_LOG" && return 0
        kill -0 $H_PID 2>/dev/null || break
        sleep 0.25
    done
    echo "bar_holder never reached HOLDING:"; cat "$H_LOG"
    kill $H_PID 2>/dev/null; wait $H_PID 2>/dev/null
    return 1
}

wc_holder_stop() {
    [ -n "${H_PID:-}" ] || return 0
    $SUDO kill "$H_PID" 2>/dev/null
    wait "$H_PID" 2>/dev/null
    H_PID=""
    sleep 0.5   # munmap frees the PAT entry before the caller reloads
    cat "$H_LOG" 2>/dev/null
}

#
# Load phoenixfs while a WC mapping covers the GPU BAR:
#   whole BAR  -> the load must FAIL cleanly (PAT conflict), and the failure
#                 must be the PAT path, NOT "refusing ... already has struct
#                 pages" pointing at phoenixfs's own p2pdma slice (4ebe13f);
#                 STAGING may instead succeed via the recovered slice.
#   window     -> FULL must load and route around the conflicted block.
# Afterwards the holder is released and the same mode must load again.
# Usage: wc_case_block <full|staging> [offMiB lenMiB]
wc_case_block() {
    local mode=$1 window_off=${2:-} window_len=${3:-}
    local whole=$([ -z "$window_off" ] && echo 1 || echo 0)
    local bdf range start size ins_rc dlog
    bdf=$(gpu_bdf) || { echo "cannot resolve GPU $GPU BDF"; return 1; }
    range=$(gpu_bar_range "$bdf"); read -r start size <<<"$range"
    [ -n "$start" ] || { echo "cannot read BAR of $bdf"; return 1; }
    dlog="$CASE_DIR/wc_dmesg.log"

    do_rmmod || return 1

    if [ -n "$window_off" ]; then
        echo "--- holder: window ${window_off}MiB+${window_len}MiB over $bdf (BAR 0x$start+0x$size)"
        wc_holder_start "$bdf" window "$window_off" "$window_len" 600 || return 1
    else
        echo "--- holder: whole BAR 0x$start+0x$size over $bdf"
        wc_holder_start "$bdf" full 600 || return 1
    fi
    if ! pat_snapshot_has_wc "$start" "$size"; then
        echo "holder created no PAT conflict over the BAR (vacuous test)"
        wc_holder_stop; return 1
    fi
    echo "PAT snapshot now has a non-write-back entry over the BAR, as intended"

    $SUDO dmesg 2>/dev/null | wc -l >"$CASE_DIR/wc_dmesg_k0"
    # debug on: the assertion below keys off phxfs_info messages
    do_insmod "$mode" phxfs_debug=1 >"$CASE_DIR/wc_insmod.log" 2>&1
    ins_rc=$?
    $SUDO dmesg 2>/dev/null | tail -n +"$(($(cat "$CASE_DIR/wc_dmesg_k0") + 1))" >"$dlog"

    if [ "$ins_rc" -eq 0 ]; then
        if [ "$whole" = 0 ]; then
            echo "FULL loaded, routing around the ${window_len}MiB WC window"
        else
            echo "loaded over the whole-BAR WC holder: this kernel tolerates remapping over a foreign memtype"
        fi
    else
        mod_loaded && { echo "insmod failed but module is loaded"; wc_holder_stop; return 1; }
        if grep -q "already has struct pages" "$dlog"; then
            echo "REGRESSION: refused via the foreign-pgmap check (a pure WC holder must not trigger it)"
            grep -m2 "refusing" "$dlog"
            wc_holder_stop; return 1
        fi
        if ! grep -qE "p2pdma bootstrap failed|devm_memremap_pages failed|all segment remaps failed" "$dlog"; then
            echo "failed via an unexpected path:"; tail -5 "$dlog"
            wc_holder_stop; return 1
        fi
        echo "load refused cleanly by the kernel's own memtype/memory-add check (expected on kernels that enforce PAT):"
        grep -m3 -E "trying the first block|bootstrap failed|devm_memremap_pages failed|all segment remaps failed|npu_ctrl_init error" "$dlog"
    fi

    wc_holder_stop || return 1
    do_rmmod || return 1     # staging-via-slice path may have loaded
    do_insmod "$mode" || return 1
    check_load "$mode" || return 1
    echo "reloaded $mode cleanly after the holder released the BAR"
}

builtin_run() {   # mode-name, "@name args..."
    local mode=$1; shift
    local name=$1; shift
    case "$name" in
    @check_load)
        check_load "$mode" ;;
    @refcnt_zero)
        wait_refcnt_zero 20 && echo "refcnt=0" ;;
    @reload)
        local n=${1:-1}; shift || true
        local i
        for i in $(seq 1 "$n"); do
            echo "--- reload $i/$n ($mode $*)"
            do_rmmod || return 1
            ! mod_loaded || { echo "still loaded after rmmod"; return 1; }
            do_insmod "$mode" "$@" || return 1
            check_load "$mode" || return 1
        done ;;
    @switch_modes)
        local n=${1:-1} i m
        for i in $(seq 1 "$n"); do
            for m in full staging; do
                echo "--- round $i: -> $m"
                do_rmmod || return 1
                do_insmod "$m" || return 1
                check_load "$m" || return 1
                echo "\$ test_ioctl $GPU basic"
                timeout -k 15 300 "$BIN/test_ioctl" "$GPU" basic || return 1
            done
        done ;;
    @double_insmod)
        echo "\$ insmod (second time)"
        if $SUDO insmod "$KO" phxfs_map_mode="$(mode_num "$mode")" 2>&1; then
            echo "second insmod unexpectedly succeeded"; return 1
        fi
        check_load "$mode" ;;
    @bad_load)
        do_rmmod || return 1
        refresh_pat
        echo "\$ insmod $KO $*   (must fail)"
        if $SUDO insmod "$KO" "$@" 2>&1; then
            echo "insmod unexpectedly succeeded"; do_rmmod; return 1
        fi
        if mod_loaded; then echo "module left loaded"; return 1; fi
        echo "rejected as expected" ;;
    @wc_hold_block)
        wc_case_block "$1" ;;
    @wc_partial)
        wc_case_block full "$1" "$2" ;;
    @rmmod_busy)
        local log="$CASE_DIR/holder.log" pid i ok=0
        "$BIN/test_ioctl" "$GPU" hold 120 >"$log" 2>&1 &
        pid=$!
        for i in $(seq 1 120); do
            grep -q HOLDING "$log" && { ok=1; break; }
            kill -0 $pid 2>/dev/null || break
            sleep 0.5
        done
        cat "$log"
        if [ $ok -ne 1 ]; then kill -9 $pid 2>/dev/null; wait $pid; echo "holder never mapped"; return 1; fi
        echo "holder pid=$pid refcnt=$(refcnt)"
        echo "\$ rmmod phoenixfs   (must fail while mapped)"
        if $SUDO rmmod phoenixfs 2>&1; then
            echo "rmmod succeeded with a live mapping"; kill -9 $pid; wait $pid; return 1
        fi
        mod_loaded || { echo "module gone"; return 1; }
        echo "\$ kill -9 holder"
        kill -9 $pid; wait $pid 2>/dev/null
        wait_refcnt_zero 40 || return 1
        do_rmmod || return 1
        do_insmod "$mode" || return 1
        check_load "$mode" ;;
    *)
        echo "unknown builtin $name"; return 2 ;;
    esac
}

# ------------------------------------------------------------------ runner

[ -n "$OUT" ] || OUT=/tmp/phx_matrix_$(date +%Y%m%d_%H%M%S)
R_ID=(); R_ST=(); R_SEC=(); R_NOTE=()
BLOCKED_full=""; BLOCKED_staging=""; BLOCKED_none=""
ANY_FAIL=0
HOLDER_PIDS=""

record() {   # id status seconds note
    R_ID+=("$1"); R_ST+=("$2"); R_SEC+=("$3"); R_NOTE+=("$4")
    local col=$C_GRN
    case "$2" in PASS) ;; XFAIL|XPASS|SKIP) col=$C_YEL ;; *) col=$C_RED; ANY_FAIL=1 ;; esac
    printf '%s[%-7s]%s %-28s %5ss  %s\n' "$col" "$2" "$C_OFF" "$1" "$3" "$4"
    [ "$DRY" -eq 1 ] || printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >>"$OUT/summary.tsv"
}

# Kernel-side verdict for the window [k0, now): prints a note, returns 1 on problems.
kernel_verdict() {   # k0 t0 dmesg_file
    local k0=$1 t0=$2 f=$3 newt bad
    $SUDO dmesg 2>/dev/null | tail -n +"$((k0 + 1))" >"$f"
    bad=$(grep -E "$BAD_RE" "$f" | head -3 | cut -c1-120)
    newt=$(( ($(taint) & ~t0) & TAINT_MASK ))
    KNOTE=""
    [ -z "$bad" ] && [ $newt -eq 0 ] && return 0
    KNOTE="kernel:"
    [ -z "$bad" ] || KNOTE+=" ${bad//$'\n'/ | }"
    [ $newt -eq 0 ] || KNOTE+=" new-taint=$newt"
    return 1
}

# Bring the module into `want` state; records a load.* result on transitions.
ensure_mode() {
    local want=$1 have k0 t0 t s rc=0 dir
    [ "$want" = keep ] && return 0
    if [ $DRY -eq 1 ]; then
        [ "$DRY_MODE" = "$want" ] || echo "   ~ switch module: $DRY_MODE -> $want"
        DRY_MODE=$want
        return 0
    fi
    have=$(cur_mode)
    [ "$have" = "$want" ] && return 0
    local blk="BLOCKED_$want"
    [ -z "${!blk}" ] || return 1
    dir="$OUT/_transition_$(printf '%03d' "$SEQ")_${have}_to_${want}"
    mkdir -p "$dir"
    k0=$(klines); t0=$(taint); s=$SECONDS
    {
        do_rmmod || rc=1
        if [ $rc -eq 0 ] && [ "$want" != none ]; then
            do_insmod "$want" || rc=1
            [ $rc -ne 0 ] || check_load "$want" || rc=1
        fi
    } >"$dir/output.log" 2>&1
    t=$((SECONDS - s))
    kernel_verdict "$k0" "$t0" "$dir/dmesg.log" || rc=2
    if [ $rc -eq 0 ]; then
        record "~switch:$have->$want" PASS "$t" ""
        return 0
    fi
    record "~switch:$have->$want" "$([ $rc -eq 2 ] && echo KFAIL || echo FAIL)" "$t" \
        "${KNOTE:-see $dir/output.log}"
    [ $rc -eq 2 ] && return 0   # loaded, but the kernel complained: keep going
    printf -v "$blk" '%s' 1
    return 1
}

run_case() {   # index
    local i=$1 id=${IDS[$1]} kmod=${KMODS[$1]} env=${ENVS[$1]} tmo=${TMOS[$1]} exp=${EXPS[$1]}
    local cmd; cmd=$(expand "${CMDS[$1]}")
    SEQ=$((SEQ + 1))
    if [ $DRY -eq 1 ]; then
        ensure_mode "$kmod"
        printf '   %-28s [%-7s] env=%s  %s\n' "$id" "$DRY_MODE" "$env" "$cmd"
        case "$cmd" in @switch_modes*) DRY_MODE=staging ;; @bad_load*) DRY_MODE=none ;; esac
        return 0
    fi
    if ! ensure_mode "$kmod"; then
        record "$id" BLOCKED 0 "module could not be brought to '$kmod'"
        return 0
    fi
    local mode; mode=$(cur_mode)
    CASE_DIR="$OUT/$(printf '%03d' "$SEQ")_$id"
    mkdir -p "$CASE_DIR"
    local envargs=()
    [ "$env" = "-" ] || IFS=',' read -r -a envargs <<<"$env"
    {
        echo "# id=$id kmod=$kmod (now: $mode, refcnt=$(refcnt)) env=$env timeout=$tmo expect=$exp"
        echo "# cmd: $cmd"
        echo "# start: $(date '+%F %T')"
    } >"$CASE_DIR/output.log"

    local k0 t0 s rc
    k0=$(klines); t0=$(taint); s=$SECONDS
    if [[ "$cmd" == @* ]]; then
        # shellcheck disable=SC2086
        ( builtin_run "$mode" $cmd ) >>"$CASE_DIR/output.log" 2>&1
        rc=$?
    else
        env ${envargs[@]+"${envargs[@]}"} timeout -k 15 "$tmo" bash -c "$cmd" >>"$CASE_DIR/output.log" 2>&1
        rc=$?
    fi
    local t=$((SECONDS - s))
    echo "# exit=$rc after ${t}s" >>"$CASE_DIR/output.log"

    local out="$CASE_DIR/output.log" st note=""
    local nfail nskip
    nfail=$(grep -c '\[FAIL\]' "$out")
    # test_io reports a size whose regmem failed as "  <size>  SKIP (regmem failed)".
    nskip=$(grep -cE '\[SKIP\]| SKIP \(' "$out")
    if [ $rc -eq 124 ] || [ $rc -eq 137 ] && [[ "$cmd" != @* ]]; then
        st=TIMEOUT; note="killed after ${tmo}s (possible kernel hang; see dmesg.log)"
    else
        case "$exp" in
            pass)   [ $rc -eq 0 ] && [ "$nfail" -eq 0 ] && st=PASS || st=FAIL ;;
            strict) if [ $rc -eq 0 ] && [ "$nfail" -eq 0 ] && [ "$nskip" -eq 0 ]; then st=PASS
                    elif [ $rc -eq 0 ] && [ "$nfail" -eq 0 ]; then st=FAIL; note="unexpected [SKIP]"
                    else st=FAIL; fi ;;
            skip)   [ $rc -eq 0 ] && [ "$nskip" -gt 0 ] && st=PASS || st=FAIL ;;
            fail)   [ $rc -ne 0 ] && st=PASS || st=FAIL ;;
            xfail)  if [ $rc -eq 0 ] && [ "$nfail" -eq 0 ]; then st=XPASS; note="known failure now passes: update matrix"
                    else st=XFAIL; note="known: $(grep -m1 '\[FAIL\]' "$out" | cut -c1-90)"; fi ;;
        esac
        if [ "$st" = FAIL ] && [ -z "$note" ]; then
            note="exit=$rc fail=$nfail skip=$nskip: $(grep -m1 -E '\[FAIL\]|FATAL|error|failed' "$out" | cut -c1-90)"
        fi
    fi
    if ! kernel_verdict "$k0" "$t0" "$CASE_DIR/dmesg.log"; then
        [ "$st" = PASS ] && st=KFAIL
        note="${KNOTE}${note:+; $note}"
    fi
    [ -s "$CASE_DIR/dmesg.log" ] || rm -f "$CASE_DIR/dmesg.log"
    record "$id" "$st" "$t" "$note"
}

cleanup() {
    local p
    for p in $(jobs -p); do kill -9 "$p" 2>/dev/null; done
}
trap 'cleanup; echo; warn "interrupted"; summary; exit 1' INT TERM

summary() {
    [ $DRY -eq 1 ] && return
    local i pass=0 fail=0 known=0
    echo
    echo "=================== summary ($OUT) ==================="
    for i in "${!R_ID[@]}"; do
        case "${R_ST[$i]}" in
            PASS) pass=$((pass + 1)) ;;
            XFAIL|XPASS|SKIP) known=$((known + 1))
                  printf '  %-8s %-28s %s\n' "${R_ST[$i]}" "${R_ID[$i]}" "${R_NOTE[$i]}" ;;
            *) fail=$((fail + 1)); printf '  %-8s %-28s %s\n' "${R_ST[$i]}" "${R_ID[$i]}" "${R_NOTE[$i]}" ;;
        esac
    done
    echo "  passed=$pass  failed=$fail  known(xfail/xpass)=$known"
    echo "  FAIL=wrong result  KFAIL=passed but the kernel logged BUG/WARN/taint"
    echo "  TIMEOUT=killed (possible kernel hang)  BLOCKED=module could not be loaded in that mode"
    echo "  per-case logs: $OUT/<nnn>_<id>/{output,dmesg}.log ; table: $OUT/summary.tsv"
}

# ------------------------------------------------------------------ preflight

needs_module=0
for k in "${KMODS[@]}"; do case "$k" in full|staging|keep) needs_module=1 ;; esac; done

if [ $DRY -eq 0 ]; then
    [ -z "$SUDO" ] || $SUDO true 2>/dev/null || die "need root or passwordless sudo"
    mkdir -p "$OUT" || die "cannot create $OUT"
    : >"$OUT/summary.tsv"
    for c in "${CMDS[@]}"; do
        b=$(expand "$c"); b=${b%% *}
        [[ "$b" == @* ]] || [ -x "$b" ] || die "missing test binary: $b (build first)"
    done
    for c in "${CMDS[@]}"; do
        case "$c" in @wc_hold_block*|@wc_partial*)
            [ -x "$BIN/bar_holder" ] || die "missing $BIN/bar_holder (build the bar_holder target)" ;;
        esac
    done
    if [ $needs_module -eq 1 ]; then
        [ -f "$KO" ] || die "module not found: $KO"
        if [ "$(cat /sys/module/module/parameters/sig_enforce 2>/dev/null)" = Y ] &&
           [ -z "$(modinfo -F signer "$KO" 2>/dev/null)" ]; then
            die "module.sig_enforce=1 but $KO is unsigned; sign a copy and pass --ko"
        fi
    fi
    mkdir -p "$(dirname "$FILE")" 2>/dev/null
    [ -w "$(dirname "$FILE")" ] || die "data directory not writable: $(dirname "$FILE")"
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    [ -z "$busy" ] || warn "GPU compute processes running (BAR must be idle for insmod):"$'\n'"$busy"
    {
        echo "date:   $(date '+%F %T')"
        echo "kernel: $(uname -r)"
        echo "repo:   $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' +dirty'))"
        echo "ko:     $KO ($(modinfo -F srcversion "$KO" 2>/dev/null), signer: $(modinfo -F signer "$KO" 2>/dev/null))"
        echo "gpu:    $GPU / $GPU2   file: $FILE"
        echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        echo "taint:  $(taint)"
    } | tee "$OUT/env.txt"
fi

info "matrix: $MATRIX (${#IDS[@]} case(s)), module now: $(cur_mode)"
SEQ=0
DRY_MODE=$(cur_mode)
for i in "${!IDS[@]}"; do
    run_case "$i"
    if [ $STOP -eq 1 ] && [ $ANY_FAIL -ne 0 ]; then
        warn "stopping at first failure"
        break
    fi
done

if [ $DRY -eq 0 ] && [ $KEEP -eq 0 ] && mod_loaded; then
    info "unloading phoenixfs"
    do_rmmod >"$OUT/final_rmmod.log" 2>&1 || warn "final rmmod failed (see $OUT/final_rmmod.log)"
fi
summary
exit $ANY_FAIL

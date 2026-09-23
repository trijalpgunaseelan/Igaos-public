#!/usr/bin/env bash
# Fetch the public benchmark instances the harnesses in this directory expect.
#
#   ./bench/fetch_benchmarks.sh [destination]      # default: ./benchmarks
#
# The instances themselves are not redistributed with this repository — they
# belong to the collections named below and are ~230 MB. Everything downloaded
# here is publicly available.
#
#   Netlib LP        114 feasible + 28 infeasible linear programs, the standard
#                    correctness test for an LP code since 1985.
#                    https://www.netlib.org/lp/   (mirror: github.com/SkyLiu0/NETLIB)
#   MIPLIB family    mixed-integer instances that ship as test data with HiGHS.
#                    https://miplib.zib.de/
#   Maros-Meszaros   138 convex quadratic programs, the standard public QP test
#                    set and the collection Mittelmann's convex QP benchmark is
#                    built on. Original: http://www.doc.ic.ac.uk/~im/
#                    (mirror: github.com/YimingYAN/QP-Test-Problems)
#
# Reference objective values for the Netlib set are in bench/netlib_reference.csv
# (produced by Gurobi 10 at 1e-8; one entry is corrected in bench/netlib.py, with
# the reason stated there).

set -u
DEST="${1:-benchmarks}"
NETLIB_RAW="https://raw.githubusercontent.com/SkyLiu0/NETLIB"
HIGHS_RAW="https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances"
HERE="$(cd "$(dirname "$0")" && pwd)"

MAROS_RAW="https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files"
mkdir -p "$DEST/netlib" "$DEST/netlib_infeasible" "$DEST/milp" "$DEST/qp_maros" "$DEST/standard"

fetch() {   # fetch <url> <path>  — skip what is already there
    [ -s "$2" ] && return 0
    curl -sfL --max-time 120 -o "$2" "$1" || { rm -f "$2"; return 1; }
}

echo "Netlib LP — feasible (114 instances, ~230 MB)"
n=0; miss=0
while IFS=, read -r name rest; do
    [ "$name" = "name" ] && continue
    [ -z "$name" ] && continue
    if fetch "$NETLIB_RAW/main/feasible/$name.mps" "$DEST/netlib/$name.mps"; then
        n=$((n+1)); printf '\r  %3d downloaded' "$n"
    else
        miss=$((miss+1)); echo; echo "  could not fetch $name"
    fi
done < "$HERE/netlib_reference.csv"
echo

echo "Netlib LP — infeasible (28 instances)"
i=0
for name in bgdbg1 bgetam bgindy bgprtr box1 ceria3d chemcom cplex1 cplex2 ex72a \
            ex73a forest6 galenet gosh gran itest2 itest6 klein1 klein2 klein3 \
            mondou2 pang pilot4i qual reactor refinery vol1 woodinfe; do
    fetch "$NETLIB_RAW/main/infeasible/$name.mps" "$DEST/netlib_infeasible/$name.mps" \
        && i=$((i+1))
done
echo "  $i downloaded"

echo "MIPLIB-family mixed-integer instances"
k=0
for name in flugpl egout gt2 lseu p0548 rgn bell5 dcmulti gesa2; do
    fetch "$HIGHS_RAW/$name.mps" "$DEST/milp/$name.mps" && k=$((k+1))
done
echo "  $k downloaded"

# The commercial comparison (bench/commercial.py) needs its own directory: the
# free CPLEX and Gurobi licences are capped at 1000x1000 and 2000x2000, so the
# set is chosen to be small enough that both licences can actually load it, and
# mixed LP/MIP so the comparison covers both paths.  Until this section existed
# the command printed in bench/results_commercial.txt named a directory nothing
# created -- the result was real, the reproduce line was not.  Defect 22.
echo "Commercial-comparison set (22 instances, small enough for the free licences)"
c=0
for name in 25fv47 adlittle afiro avgas bell5 dcmulti e226 egout etamacro gesa2 \
            gt2 israel lseu p0548 perold rgn scrs8 shell stair standata \
            standgub standmps; do
    fetch "$HIGHS_RAW/$name.mps" "$DEST/standard/$name.mps" && c=$((c+1))
done
echo "  $c downloaded"

echo "Maros-Meszaros convex QP (138 instances, ~220 MB)"
q=0
for name in AUG2D AUG2DC AUG2DCQP AUG2DQP AUG3D AUG3DC AUG3DCQP AUG3DQP BOYD1 BOYD2 \
    CONT-050 CONT-100 CONT-101 CONT-200 CONT-201 CONT-300 CVXQP1_L CVXQP1_M CVXQP1_S \
    CVXQP2_L CVXQP2_M CVXQP2_S CVXQP3_L CVXQP3_M CVXQP3_S DPKLO1 DTOC3 DUAL1 DUAL2 \
    DUAL3 DUAL4 DUALC1 DUALC2 DUALC5 DUALC8 EXDATA GENHS28 GOULDQP2 GOULDQP3 HS118 \
    HS21 HS268 HS35 HS35MOD HS51 HS52 HS53 HS76 HUES-MOD HUESTIS KSIP LASER LISWET1 \
    LISWET10 LISWET11 LISWET12 LISWET2 LISWET3 LISWET4 LISWET5 LISWET6 LISWET7 \
    LISWET8 LISWET9 LOTSCHD MOSARQP1 MOSARQP2 POWELL20 PRIMAL1 PRIMAL2 PRIMAL3 \
    PRIMAL4 PRIMALC1 PRIMALC2 PRIMALC5 PRIMALC8 Q25FV47 QADLITTL QAFIRO QBANDM \
    QBEACONF QBORE3D QBRANDY QCAPRI QE226 QETAMACR QFFFFF80 QFORPLAN QGFRDXPN \
    QGROW15 QGROW22 QGROW7 QISRAEL QPCBLEND QPCBOEI1 QPCBOEI2 QPCSTAIR QPILOTNO \
    QPTEST QRECIPE QSC205 QSCAGR25 QSCAGR7 QSCFXM1 QSCFXM2 QSCFXM3 QSCORPIO QSCRS8 \
    QSCSD1 QSCSD6 QSCSD8 QSCTAP1 QSCTAP2 QSCTAP3 QSEBA QSHARE1B QSHARE2B QSHELL \
    QSHIP04L QSHIP04S QSHIP08L QSHIP08S QSHIP12L QSHIP12S QSIERRA QSTAIR QSTANDAT \
    S268 STADAT1 STADAT2 STADAT3 STCQP1 STCQP2 TAME UBH1 VALUES YAO ZECEVIC2; do
    fetch "$MAROS_RAW/$name.QPS" "$DEST/qp_maros/$name.QPS" \
        && { q=$((q+1)); printf '\r  %3d downloaded' "$q"; }
done
echo

cat <<EOF

Done. Now run:

  IGAOS_BIN=./build/igaos python3 bench/netlib.py \\
      --dir $DEST/netlib --ref bench/netlib_reference.csv --time-limit 300

  IGAOS_BIN=./build/igaos python3 bench/infeasible.py --dir $DEST/netlib_infeasible

  IGAOS_BIN=./build/igaos python3 bench/mps_bench.py --dir $DEST/milp --time-limit 300

  pip install osqp
  IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir $DEST/qp_maros --time-limit 20

  pip install cplex gurobipy
  python3 bench/commercial.py --dir $DEST/standard --time-limit 60
EOF
[ "$miss" -gt 0 ] && echo "($miss feasible instances could not be fetched)"
exit 0

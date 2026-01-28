methodname="Dimer"
echo "Total converged: $(cat ${methodname}_status_csvs/status_rank_*.csv | grep -c ",converged") / $(cat ${methodname}_status_csvs/status_rank_*.csv | wc -l)"
echo "Total not converged: $(cat ${methodname}_status_csvs/status_rank_*.csv | grep -c ",not_converged") / $(cat ${methodname}_status_csvs/status_rank_*.csv | wc -l)"
echo "Total stopped early: $(cat ${methodname}_status_csvs/status_rank_*.csv | grep -c ",not_converged_StopRun") / $(cat ${methodname}_status_csvs/status_rank_*.csv | wc -l)"
for i in {0..15}; do echo "Rank $i: $(grep -c ",converged" Dimer_status_csvs/status_rank_$i.csv) / $(wc -l < Dimer_status_csvs/status_rank_$i.csv)"; done


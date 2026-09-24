#!/usr/bin/env bash
# Print free memory and running python / tsp jobs. Run before queueing anything heavy.
set -uo pipefail
if [[ "$(uname)" == "Darwin" ]]; then
  total=$(sysctl -n hw.memsize)
  page=$(sysctl -n hw.pagesize)
  vm=$(vm_stat)
  free=$(echo "$vm" | awk -v p="$page" '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); print (f+i+s)*p}')
  swap=$(sysctl -n vm.swapusage | awk '{print $6" used"}')
  printf "memory: %.1f GB free of %.0f GB, swap %s\n" "$(echo "$free/1073741824" | bc -l)" "$(echo "$total/1073741824" | bc -l)" "$swap"
else
  free -h | awk 'NR==2{print "memory: "$7" free of "$2}'
fi
echo "python processes:"
ps -eo pid,rss,etime,args | awk 'NR==1 || ($4 ~ /python/ && !/awk/)' | awk 'NR==1{print "  "$0; next}{printf "  pid %s rss %.2f GB up %s  %s\n",$1,$2/1048576,$3,substr($0,index($0,$4),80)}'
echo "tsp queue:"
TSP="$(command -v tsp || command -v ts || true)"
[[ -n "$TSP" ]] && "$TSP" -l 2>/dev/null | sed 's/^/  /' || echo "  tsp not running"

#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim/bin/python -m smartsim._core.entrypoints.colocated +ifname lo +lockfile smartsim-6e7b380.lock +db_cpus 4 +command /lus/grand/projects/datascience/ashao/local/thetagpu/smartsim-deps/gcc-9.3.0/bin/redis-server /lus/grand/projects/datascience/ashao/local/thetagpu/smartsim-deps/gcc-9.3.0/config/redis6.conf --loadmodule /lus/grand/projects/datascience/ashao/local/thetagpu/smartsim-deps/gcc-9.3.0/lib/redisai.so --port 6780 --logfile /dev/null &
DBPID=$!

$@


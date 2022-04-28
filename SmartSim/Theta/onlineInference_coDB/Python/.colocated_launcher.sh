#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

/projects/cfdml_aesp/balin/SmartSim/ssim/bin/python -m smartsim._core.entrypoints.colocated +ifname lo +lockfile smartsim-ec4c0e0.lock +db_cpus 32 +command /lus/theta-fs0/projects/cfdml_aesp/balin/SmartSim/smartsim-develop/smartsim/_core/bin/redis-server /lus/theta-fs0/projects/cfdml_aesp/balin/SmartSim/smartsim-develop/smartsim/_core/config/redis6.conf --loadmodule /lus/theta-fs0/projects/cfdml_aesp/balin/SmartSim/smartsim-develop/smartsim/_core/lib/redisai.so --port 6780 --logfile /dev/null &
DBPID=$!

taskset -c 0-$(nproc --ignore=33) $@


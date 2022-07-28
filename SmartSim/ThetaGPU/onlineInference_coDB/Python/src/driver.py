# general imports
import os
import sys

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import MpirunSettings

# Parse command line arguments
nodes = int(sys.argv[1])
ppn = int(sys.argv[2])
simprocs = int(sys.argv[3])
simprocs_pn = int(sys.argv[4])
dbprocs_pn = int(sys.argv[5])

# Initialize the SmartSim Experiment
PORT = 6780
exp = Experiment("inference-example", launcher="cobalt")

# Set the run settings, including the Python executable and how to run it
Py_exe = './src/inference.py'
if (simprocs_pn<ppn):
    ppn = simprocs_pn
runArgs = {'n': simprocs, 'N': ppn}
exe_args = Py_exe+ f' --dbnodes=1'
exp_settings = exp.create_run_settings(
    exe='python',
    exe_args=exe_args,
    run_command='mpirun',
    run_args=runArgs,
    env_vars=None
)

# Create and start the co-located database model
colo_model = exp.create_model("inference", exp_settings)
#kwargs = {
#    maxclients: 100000,
#    threads_per_queue: 1, # set to 4 for improved performance
#    inter_op_threads: 1,
#    intra_op_threads: 1,
#    server_threads: 2 # keydb only
#}
colo_model.colocate_db(
        port=PORT,              # database port
        db_cpus=dbprocs_pn,     # cpus given to the database on each node
        debug=False,            # include debug information (will be slower)
        limit_app_cpus=False,    # limit the number of cpus used by the app
        ifname='lo',            # specify network interface to use (i.e. "ib0")
)
exp.start(colo_model, summary=True)

# Quit
print("Done")
print("Quitting")


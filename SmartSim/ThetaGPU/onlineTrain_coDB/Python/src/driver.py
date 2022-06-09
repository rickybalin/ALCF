# general imports
import os
import sys

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import MpirunSettings

# Define function to parse node list
def parseNodeList(fname):
    #fname = os.environ['COBALT_NODEFILE']
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes

# Parse command line arguments
nodes = int(sys.argv[1])
ppn = int(sys.argv[2])
simprocs = int(sys.argv[3])
simprocs_pn = int(sys.argv[4])
mlprocs = int(sys.argv[5])
mlprocs_pn = int(sys.argv[6])
dbprocs_pn = int(sys.argv[7])
device = sys.argv[8]
hostfile = sys.argv[9]

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nNodes = parseNodeList(fname)
print(f"\nRunning on {nNodes} total nodes on ThetaGPU")
print(nodelist, "\n")
hosts = ','.join(nodelist)

# Initialize the SmartSim Experiment
PORT = 6780
exp = Experiment("train-example", launcher="cobalt")

# Set the run settings, including the Python executable and how to run it
Py_exe = './src/load_data.py'
if (simprocs_pn<ppn):
    ppn = simprocs_pn
runArgs = {'hostfile': hostfile,
           'n': simprocs, 'npernode': ppn,
           'x': 'PATH'
          }
exe_args = Py_exe+ f' --dbnodes 1 --ppn {simprocs_pn}'
exp_settings = exp.create_run_settings(
    exe='python',
    exe_args=exe_args,
    run_command='mpirun',
    run_args=runArgs,
    env_vars=None
)

# Create and start the co-located database model
colo_model = exp.create_model("load_data", exp_settings)
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
exp.start(colo_model, block=False, summary=True)

# Python data consumer (i.e. training)
print("Launching data consumer ...")
if (device=='cuda' and mlprocs_pn>8):
    ppn = 8
else:
    ppn = mlprocs_pn
runArgs = {'hostfile': hostfile,
           'n': mlprocs, 'npernode': ppn,
           'x': 'PATH'
          }
ml_exe = f"src/trainPar.py --device {device} --ppn {ppn}"
runML = MpirunSettings("python", 
        exe_args=ml_exe, 
        run_args=runArgs
        )
ml_model = exp.create_model("train_model", runML)
exp.start(ml_model, block=True, summary=False)
print("Done\n")

# Quit
print("Done")
print("Quitting")


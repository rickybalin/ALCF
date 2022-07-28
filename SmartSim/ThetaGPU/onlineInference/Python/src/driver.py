import os, sys, time
from smartsim import Experiment
from smartsim.settings import MpirunSettings

# Parse command line arguments
ppn = int(sys.argv[1]) # CPU cores per node
nodes = int(sys.argv[2])
db_nNodes = int(sys.argv[3])
sim_nNodes = int(sys.argv[4])
simprocs = int(sys.argv[5])

# Define function to parse node list
def parseNodeList():
    fname = os.environ['COBALT_NODEFILE']
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes
    
# Get nodes of this allocation (job) and split them between the tasks
nodelist, nNodes = parseNodeList()
print(f"\nRunning on {nNodes} total nodes on Theta")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: db_nNodes])
simNodes = ','.join(nodelist[db_nNodes: db_nNodes + sim_nNodes])
print(f"Database running on {db_nNodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {sim_nNodes} nodes:")
print(simNodes)
print("")

# Set up database and start it
PORT = 6780
exp = Experiment("train-example", launcher="cobalt")
db = exp.create_database(port=PORT, batch=False, db_nodes=db_nNodes,
                         run_args={"node-list": dbNodes, "cpus-per-pe": ppn})
print("Starting database ...")
exp.start(db)
print("Done\n")

# Python inference routine
print("Launching Python inference routine ...")
Py_exe = './src/inference.py'
exe_args = Py_exe+ f' --dbnodes={db_nNodes}'
mpirun = MpiprunSettings('python',
        exe_args=exe_args, 
        run_args={'node-list': simNodes})
mpirun.set_tasks(simprocs)
if (simprocs >= ppn):
    mpirun.set_tasks_per_node(ppn)
else:
    mpirun.set_tasks_per_node(simprocs)
inf_data = exp.create_model("inference", mpirun)
exp.start(inf_data, summary=False, block=True)
print("Done\n")

# Stop database
print("Stopping the Orchestrator ...")
exp.stop(db)
print("Done")
print("Quitting")


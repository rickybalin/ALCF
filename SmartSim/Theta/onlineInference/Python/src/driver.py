import os, sys, time
from smartsim import Experiment
from smartsim.settings import AprunSettings

# Parse command line arguments
ppn = int(sys.argv[1])
nodes = int(sys.argv[2])
dbnodes = int(sys.argv[3])
simnodes = int(sys.argv[4])
simprocs = int(sys.argv[5])

# Define function to parse node list
def parseNodeList():
    cobStr = os.environ['COBALT_PARTNAME']
    tmp = cobStr.split(',')
    nodelist = []
    for item in tmp:
        if (item.find('-') > 0):
            tmp2 = item.split('-')
            istart = int(tmp2[0])
            iend = int(tmp2[1])
            for i in range(istart,iend+1):
                nodelist.append(str(i))
        else:
            nodelist.append(item)
    nnodes = len(nodelist)
    return nodelist, nnodes

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nnodes = parseNodeList()
print(f"\nRunning on {nnodes} total nodes on Theta")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: dbnodes])
simNodes = ','.join(nodelist[dbnodes: dbnodes + simnodes])
print(f"Database running on {dbnodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {simnodes} nodes:")
print(simNodes)
print("")

# Set up database and start it
PORT = 6780
exp = Experiment("train-example", launcher="cobalt")
db = exp.create_database(port=PORT, batch=False, db_nodes=dbnodes,
                         run_args={"node-list": dbNodes, "cpus-per-pe": 64})
print("Starting database ...")
exp.start(db)
print("Done\n")

# Python inference routine
print("Launching Python inference routine ...")
Py_exe = './src/inference.py'
exe_args = Py_exe+ f' --dbnodes={dbnodes}'
aprun = AprunSettings('python',
        exe_args=exe_args, 
        run_args={'node-list': simNodes})
aprun.set_tasks(simprocs)
if (simprocs >= ppn):
    aprun.set_tasks_per_node(ppn)
else:
    aprun.set_tasks_per_node(simprocs)
inf_data = exp.create_model("inference", aprun)
exp.start(inf_data, summary=False, block=True)
print("Done\n")


# Stop database
print("Stopping the Orchestrator ...")
exp.stop(db)
print("Done")
print("Quitting")


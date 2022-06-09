# general imports
import os
import sys

# Define function to parse node list
def parseNodeList():
    fname = os.environ['COBALT_NODEFILE']
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
    nNodes = len(nodelist)
    return fname, nodelist, nNodes

# Get nodes of this allocation (job) and split them between the tasks
hostfile, nodelist, nNodes = parseNodeList()
print(f"\nRunning on {nNodes} total nodes on ThetaGPU")
print(nodelist, "\n")
hosts = ','.join(nodelist)
print('The hosts are:')
print(hosts)
print('and the host file is:')
print(hostfile)
print('')

# Get env variables to pass to mpirun
PATH = os.getenv('PATH')
LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH')
print(PATH)
print('')
print(LD_LIBRARY_PATH)
print('')
print(':'.join((PATH,LD_LIBRARY_PATH)))

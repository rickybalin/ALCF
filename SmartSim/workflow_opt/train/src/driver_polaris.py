# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import MpiexecSettings


## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes


## Co-located DB launch
def launch_coDB(cfg, nodelist, nNodes):
    # Print nodelist
    print(f"\nRunning on {nNodes} total nodes on Polaris")
    print(nodelist, "\n")
    hosts = ','.join(nodelist)

    # Initialize the SmartSim Experiment
    PORT = cfg.experiment.port
    exp = Experiment(cfg.experiment.name, launcher=cfg.experiment.launcher)

    # Set the run settings, including the Python executable and how to run it
    Py_exe = cfg.experiment.sim_executable
    exe_args = Py_exe + f' --dbnodes 1' \
                      + f' --ppn {cfg.run_args.simprocs_pn} --logging {cfg.logging}'
    run_settings = MpiexecSettings(
                   'python',
                   exe_args=exe_args,
                   run_args=None,
                   env_vars=None
                   )
    run_settings.set_tasks(cfg.run_args.simprocs)
    run_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
    run_settings.set_hostlist(hosts)
    run_settings.set_cpu_binding_type(cfg.run_args.sim_cpu_bind)

    # Create the co-located database model
    colo_model = exp.create_model("load_data", run_settings)
    #kwargs = {
    #    maxclients: 100000,
    #    threads_per_queue: 1, # set to 4 for improved performance
    #    inter_op_threads: 1,
    #    intra_op_threads: 1,
    #    #server_threads: 2 # keydb only
    #    }
    colo_model.colocate_db(
            port=PORT,
            db_cpus=cfg.run_args.dbprocs_pn,
            debug=True,
            limit_app_cpus=False,
            ifname=cfg.run_args.network_interface,
            )

    # Start the co-located model
    exp.start(colo_model, block=False, summary=True)

    # Set up the data consumer (i.e. training)
    ml_exe = cfg.experiment.ml_executable
    exe_args = ml_exe + f'  --dbnodes 1 --device {cfg.run_args.device}' \
               + f' --ppn {cfg.run_args.mlprocs_pn} --logging {cfg.logging}'
    run_settings_ML = MpiexecSettings(
            'python', 
            exe_args=exe_args, 
            run_args=None,
            env_vars=None
            )
    print(cfg.run_args.mlprocs,cfg.run_args.mlprocs_pn,hosts,cfg.run_args.ml_cpu_bind)
    run_settings_ML.set_tasks(cfg.run_args.mlprocs)
    run_settings_ML.set_tasks_per_node(cfg.run_args.mlprocs_pn)
    run_settings_ML.set_hostlist(hosts)
    run_settings_ML.set_cpu_binding_type(cfg.run_args.ml_cpu_bind)

    # Start the data consumer
    ml_model = exp.create_model("train_model", run_settings_ML)
    exp.start(ml_model, block=True, summary=True)



## Main function
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Get nodes of this allocation (job)
    hostfile = os.getenv('PBS_NODEFILE')
    nodelist, nNodes = parseNodeList(hostfile)

    # Call appropriate launcher
    if (cfg.database.launch == "colocated"):
        launch_coDB(cfg,nodelist,nNodes)
    elif (cfg.database.launch == "clustered"):
        launch_clDB(cfg,nodelist,nNodes)
    else:
        print("\nERROR: Launcher is either colocated or clustered\n")

    # Quit
    print("Done")
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()

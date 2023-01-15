# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import PalsMpiexecSettings


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
    PORT = cfg.database.port
    exp = Experiment(cfg.experiment.name, launcher=cfg.experiment.launcher)

    # Set the run settings, including the Python executable and how to run it
    Py_exe = cfg.sim.executable
    exe_args = Py_exe + f' --dbnodes 1 --device {cfg.sim.device}' \
                      + f' --ppn {cfg.run_args.simprocs_pn} --logging {cfg.logging}'
    run_settings = PalsMpiexecSettings(
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
    colo_model = exp.create_model("inference", run_settings)
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
            ifname=cfg.database.network_interface,
            )

    # Add the ML model
    device_tag = 'CPU' if cfg.model.device=='cpu' else 'GPU'
    colo_model.add_ml_model('model',
                            cfg.model.backend,
                            model=None,  # this is used if model is in memory
                            model_path=cfg.model.path,
                            device=device_tag,
                            batch_size=cfg.model.batch,
                            min_batch_size=cfg.model.batch,
                            devices_per_node=cfg.model.devices_per_node, # only for GPU
                            inputs=None, outputs=None )

    # Start the co-located model
    exp.start(colo_model, block=True, summary=True)


## Clustered DB launch
def launch_clDB(cfg, nodelist, nNodes):
    # Split nodes between the components
    dbNodes = ','.join(nodelist[0: cfg.run_args.db_nNodes])
    dbNodes_list = nodelist[0: cfg.run_args.db_nNodes]
    simNodes = ','.join(nodelist[cfg.run_args.db_nNodes: \
                                 cfg.run_args.db_nNodes + cfg.run_args.sim_nNodes])
    print(f"Database running on {cfg.run_args.db_nNodes} nodes:")
    print(dbNodes)
    print(f"Simulatiom running on {cfg.run_args.sim_nNodes} nodes:")
    print(simNodes)
    print("")

    # Set up database and start it
    PORT = cfg.database.port
    exp = Experiment(cfg.experiment.name, launcher=cfg.experiment.launcher)
    runArgs = {"np": 1, "ppn": cfg.run_args.cores_pn}
    db = exp.create_database(port=PORT, 
                             batch=False,
                             db_nodes=cfg.run_args.db_nNodes,
                             run_command='mpiexec',
                             interface=cfg.run_args.network_interface, 
                             hosts=dbNodes_list,
                             run_args=runArgs,
                             single_cmd=False
                            )
    exp.generate(db)
    print("Starting database ...")
    exp.start(db)
    print("Done\n")

    # Python inference routine
    print("Launching Python inference routine ...")
    Py_exe = cfg.sim.executable
    exe_args = Py_exe + f' --dbnodes {cfg.run_args.db_nNodes}' \
                      + f' --device {cfg.sim.device}' \
                      + f' --ppn {cfg.run_args.simprocs}' \
                      + f' --logging {cfg.logging}'
    run_settings = PalsMpiexecSettings('python',
                  exe_args=exe_args,
                  run_args=None
                  )
    run_settings.set_tasks(cfg.run_args.simprocs)
    run_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
    run_settings.set_hostlist(simNodes)
    inf_exp = exp.create_model("inference", run_settings)
   
    # Add the ML model
    device_tag = 'CPU' if cfg.model.device=='cpu' else 'GPU'
    inf_exp.add_ml_model('model',
                         cfg.model.backend,
                         model=None,  # this is used if model is in memory
                         model_path=cfg.model.path,
                         device=device_tag,
                         batch_size=cfg.model.batch,
                         min_batch_size=cfg.model.batch,
                         devices_per_node=cfg.model.devices_per_node, # only for GPU
                         inputs=None, outputs=None )

    # Start the inference model
    exp.start(inf_exp, summary=True, block=True)
    print("Done\n")

    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)


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

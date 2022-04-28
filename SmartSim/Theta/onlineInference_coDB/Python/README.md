# Example of Online Inference with the SmartSim Co-located Database and Python Client

# Run Instructions

## Notes

- This is an example of doing online inference from a Python application with a Pytorch model using the SmartSim and SmartRedis API. 
- The main feature of this example is that the database is co-located with the Python application, sharing the CPU cores on the same nodes.
- The Pytorch model is `model_jit.pt` and it is a simple ANN trained seprately to predict a quadratic polynomial fairly well.
- Currently, the case is set up to run the Python application on 2 nodes, using 32 out of the 64 cores on each of those nodes, and the database on the remaining 32 cores of the 2 nodes. You can change the number of nodes, number of MPI processes of the Python application, and the distribution of CPU cores to each component in the submit script. The value of the nnDB variable in `src/inference.f` is always 1 in this case. You'll also note from the source code that the model needs to be uploaded to each of the databases on the allocated nodes, so rank 0 uploads the model to the database on node 1 and rank 32 does the same for the database on node 2). This is because the databases are separate and data in them (or models) are not shared. 
- The conda env needed to use the SmartSim API is specified in the `run.sh` file. Instructions on how to build the conda env for Theta are [here](../../build/README.md).
- There is a known benign issue with this example. The inference application runs fine, but the SmartSim API which launches it together with the co-located database reports a failure. I suspect it is related to the SmartSim API since it goes away by specifying `debug=True` when launching the co-located database in `src/driver.py`.

## Submit a batch job

1. Clean the run directory with `./cleanDir.sh`
2. Submit the job to the queue with `./submit.sh`. This will create a file called `JobID.cobaltlog`, where JobID will be replaced by the actual Job ID.
3. Use `tail -f JobID.cobaltlog` and `tail -f JobID.out` to monitor the progress of the job.
4. Use `tail -f inference.out` to monitor the output of the Python application.

## Submit an interactive job

1. Clean the run directory with `./cleanDir.sh`
2. Submit the interactive job to the queue with `./submit_interactive.sh`. This will create a file called `JobID.cobaltlog`, where JobID will be replaced by the actual Job ID.
3. Once the job is allowed to run, you will be placed on the MOM nodes. Change directory to the run directory. From here, run the job with `./run.sh 2 64 64 32 32`. This terminal will show the output of the SmartSim driver script.
4. In a separate terminal, log into Theta and change directory to the run directory. From here use `tail -f inference.out` to monitor the output of the Python application.


# Plot Predictions

The predictions made by the Python application will be in the `predictions_node#.dat` files, with # being replaced by the node number. Specifically, these are the predictions made by rank 0 on the first node and rank 32 on the second node used by the application. The files were separated in this way to show how inference is being performed correctly on each node.
The first column lists the x coordinate, the second column lists the predicted y values, the third column lists the true y values. 
You can visualize these results with `gnuplot` if you ssh'ed into the Theta loging nodes with `ssh -X user@alcf.anl.gov`.
You should expect to see fair agreement between the predictions and true values.

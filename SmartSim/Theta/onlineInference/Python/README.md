# Example of Online Inference with SmartSim and a Python Client

# Run Instructions

## Notes

- This is an example of doing online inference from a Python application with a Pytorch model using the SmartSim and SmartRedis API.
- The Pytorch model is `model_jit.pt` and it is a simple ANN trained seprately to predict a quadratic polynomial fairly well.
- Currently, the case is set up to run the Python application on 1 node and the Orchestrator on another node, so this is a 2 node job. One node for the Orchestrator will be plenty for this example, but you can rise this value in the submit script (avoiding 2 nodes). 
- The application is set to run with 2 MPI processes, but this can be changed in the submit scripts.
- The conda env needed to use the SmartSim API is specified in the `run.sh` file. Instructions on how to build the conda env for Theta are [here](../../build/README.md).

## Submit a batch job

1. Clean the run directory with `./cleanDir.sh`
2. Submit the job to the queue with `./submit.sh`. This will create a file called `JobID.cobaltlog`, where JobID will be replaced by the actual Job ID.
3. Use `tail -f JobID.cobaltlog` and `tail -f JobID.out` to monitor the progress of the job.
4. Use `tail -f inference.out` to monitor the output of the Python application.

## Submit an interactive job

1. Clean the run directory with `./cleanDir.sh`
2. Submit the interactive job to the queue with `./submit_interactive.sh`. This will create a file called `JobID.cobaltlog`, where JobID will be replaced by the actual Job ID.
3. Once the job is allowed to run, you will be placed on the MOM nodes. Change directory to the run directory. From here, run the job with `./run.sh 64 2 1 1 2`. This terminal will show the output of the SmartSim driver script.
4. In a separate terminal, log into Theta and change directory to the run directory. From here use `tail -f inference.out` to monitor the output of the Python application.


# Plot Predictions

The predictions made by the Python application will be in the `predictions.dat` file. Specifically, these are the predictions made by rank 0 of the application.
The first column lists the x coordinate, the second column lists the predicted y values, the third column lists the true y values. 
You can visualize these results with `gnuplot` if you ssh'ed into the Theta loging nodes with `ssh -X user@alcf.anl.gov`.
You should expect to see decent agreement between the predictions and true values.

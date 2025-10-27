
# run an interactive session on the cpu_short partition with 4GB memory per CPU for up to 3 hours
srun -p cpu_short --mem-per-cpu=4G -t 00-03:00:00  --pty bash -c

    # check if venv exists
    if [ -d "venv" ]; then
        echo "Activating existing virtual environment..."
        source venv/bin/activate
    else
        echo "Creating new virtual environment..."
        # first load python module
        module load python/gpu/3.10.6-cuda12.9
        python3 -m venv venv
        source venv/bin/activate
        echo "Installing required packages..."
        pip install -r requirements.txt
    fi
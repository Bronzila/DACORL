VERSION=extended_vel
ID=0
NUM_RUNS=1000
RESULTS_DIR=data_hetero_1k

# Create the directory if it doesn't exist
mkdir -p $RESULTS_DIR/custom_paths

for FCT in "Ackley" "Rastrigin" "Rosenbrock" "Sphere"
do
    for AGENT in "exponential_decay" "step_decay" "sgdr" "constant"    
    do
        python data_gen.py --save_run_data --save_rep_buffer --env $FCT\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    done
    # Create the JSON file with the specified content
    cat <<EOL > $RESULTS_DIR/custom_paths/$FCT.json
    [
        "$RESULTS_DIR/ToySGD/exponential_decay/0/$FCT",
        "$RESULTS_DIR/ToySGD/step_decay/0/$FCT",
        "$RESULTS_DIR/ToySGD/sgdr/0/$FCT",
        "$RESULTS_DIR/ToySGD/constant/0/$FCT"
    ]
EOL

    python combine_buffers.py --custom_paths $RESULTS_DIR/custom_paths/$FCT.json --combined_dir $RESULTS_DIR/ToySGD/combined/$FCT
done
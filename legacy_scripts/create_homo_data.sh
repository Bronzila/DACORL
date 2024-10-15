VERSION=extended_vel
IDS="0 1 2 3 4"
NUM_RUNS=1000
RESULTS_DIR=data_teacher_eval_66316748


for AGENT in "exponential_decay"
do
    for FCT in "Ackley" "Rastrigin" "Rosenbrock" "Sphere"
    do
        # for ID in $IDS
        # do
        #     python data_gen.py --save_run_data --save_rep_buffer --env $FCT\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
        # done
        python combine_buffers.py --root_dir $RESULTS_DIR/ToySGD/$AGENT --function $FCT --combined_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FCT
    done
done
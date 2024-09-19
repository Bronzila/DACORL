COMBINED_IDS="combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c"

for COMBINED_ID in $COMBINED_IDS
do
    python combine_buffers.py --custom_paths SGD_data/teacher_20_cpu_66316748/custom_paths/$COMBINED_ID.json --combined_dir SGD_data/teacher_20_cpu_66316748/SGD/$COMBINED_ID/
done
#!/bin/bash
# Submit all 4 teacher HPO jobs

echo "=========================================="
echo "Submitting DACORL Teacher HPO Jobs"
echo "=========================================="

cd jobs

echo -n "1. Constant teacher... "
JOB1=$(sbatch --parsable job_constant.sh)
echo "Job ID: $JOB1"

echo -n "2. Exponential decay teacher... "
JOB2=$(sbatch --parsable job_exponential.sh)
echo "Job ID: $JOB2"

echo -n "3. Step decay teacher... "
JOB3=$(sbatch --parsable job_step.sh)
echo "Job ID: $JOB3"

echo -n "4. SGDR teacher... "
JOB4=$(sbatch --parsable job_sgdr.sh)
echo "Job ID: $JOB4"

cd ..

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1,$JOB2,$JOB3,$JOB4"
echo ""
echo "Check output:"
echo "  tail -f jobs/DACORL-HPO-constant-${JOB1}.out"
echo "  tail -f jobs/DACORL-HPO-exponential-${JOB2}.out"
echo ""
echo "Cancel all:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4"
echo ""

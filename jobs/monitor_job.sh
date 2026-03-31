#!/bin/bash
# Monitor GPU and RAM usage for running SLURM jobs

echo "======================================"
echo "    SLURM Job Monitor"
echo "======================================"
echo ""

# Get job info
echo "📊 Your Running Jobs:"
squeue -u hasana -o "%.10i %.12P %.20j %.8T %.10M %.15R" || echo "No jobs found"
echo ""

# Get the node name and job ID
NODES=$(squeue -u hasana -h -o "%R" 2>/dev/null)
JOB_IDS=$(squeue -u hasana -h -o "%i" 2>/dev/null)

if [ -z "$NODES" ]; then
    echo "❌ No running jobs found for user: hasana"
    exit 0
fi

# Loop through each node
for NODE in $NODES; do
    echo "======================================"
    echo "🖥️  Node: $NODE"
    echo "======================================"
    
    # GPU Status
    echo ""
    echo "🎮 GPU Status:"
    ssh -o StrictHostKeyChecking=no $NODE "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits" 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s (%s):\n    Utilization: %s%%\n    Memory: %s / %s MB (%.1f%%)\n    Temp: %s°C\n    Power: %s W\n\n", 
                 $1, $2, $3, $4, $5, ($4/$5)*100, $6, $7}'
    
    # Memory usage of user processes
    echo "💾 RAM Usage (Your Processes):"
    ssh -o StrictHostKeyChecking=no $NODE "ps aux | grep 'hasana.*python' | grep -v grep" 2>/dev/null | \
    awk '{printf "  PID: %s | CPU: %s%% | RAM: %.2f GB | CMD: %s\n", $2, $3, $6/1024/1024, substr($0, index($0,$11))}'
    
    echo ""
done

echo "======================================"
echo "💡 Usage: watch -n 5 ./monitor_job.sh"
echo "   (Updates every 5 seconds)"
echo "======================================"

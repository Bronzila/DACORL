#!/usr/bin/env python3
"""Quick test script for nanoGPT HPO - runs 1 trial only"""

import sys
sys.path.insert(0, '/home/somenomus/Music/DACORL')

from teacher_hpo_SGD import Optimizee
from omegaconf import OmegaConf
from pathlib import Path

# Create minimal config
config = OmegaConf.create({
    'env': {
        'type': 'LayerwiseNanoGPT',
        'dataset_name': 'Shakespeare',
        'num_epochs': 1,  # Just 1 epoch
        'dataset_path': 'datasets',
        'initial_learning_rate': 0.0001,
        'epoch_mode': False,  # CRITICAL: respect iters_per_epoch
        'iters_per_epoch': 5,  # Just 5 iterations for testing
        'training_batch_size': 2,
        'n_layer': 6,
        'n_embd': 384,
        'block_size': 512,
        'grad_acc_steps': 10,  # Reduced for speed
        'eval_iters': 2,
        'train_validation_ratio': 0.9,
        'fraction_of_dataset': 1.0,
    },
    'teacher': 'constant',
    'results_dir': Path('./test_hpo'),
    'seed': 42,
    'dataset_path': 'datasets'
})

print("="*80)
print("QUICK HPO TEST - 1 TRIAL ONLY")
print("="*80)
print(f"\nConfig:")
print(f"  Epochs: 1")
print(f"  Iters per epoch: 5")
print(f"  Total iterations: 5")
print(f"  Grad acc steps: 10")
print(f"  Expected time: ~30 seconds")
print("\n" + "="*80)

# Create optimizee
num_seeds = 1  # Just one seed for testing
optimizee = Optimizee(config, num_seeds)

# Test one configuration
test_config = optimizee.configspace.sample_configuration()
print(f"\nTesting LR: {test_config['initial_learning_rate']:.6e}")

import time
start = time.time()

try:
    cost = optimizee.train(test_config, seed=42, budget=1)
    end = time.time()
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print(f"Cost (negative val acc): {cost:.6f}")
    print(f"Validation accuracy: {-cost*100:.2f}%")
    print(f"Time taken: {end-start:.1f} seconds")
    print("\n✓ No bugs detected!")
    print("✓ Ready for full HPO run")
    
except Exception as e:
    end = time.time()
    print("\n" + "="*80)
    print("❌ ERROR!")
    print("="*80)
    print(f"Error: {e}")
    print(f"Time before crash: {end-start:.1f} seconds")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
Verify NanoGPT Shakespeare Setup
Checks that all configs and datasets are properly configured before data collection.
"""

import json
import os
from pathlib import Path

def check_icon(condition):
    return "✅" if condition else "❌"

def main():
    print("=" * 60)
    print("NanoGPT Shakespeare Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check dataset
    print("\n📦 DATASET CHECK:")
    dataset_path = Path("datasets/shakespeare")
    train_bin = dataset_path / "train.bin"
    test_bin = dataset_path / "test.bin"
    val_bin = dataset_path / "val.bin"
    
    dataset_ok = train_bin.exists() and test_bin.exists() and val_bin.exists()
    print(f"  {check_icon(train_bin.exists())} train.bin exists")
    print(f"  {check_icon(test_bin.exists())} test.bin exists")
    print(f"  {check_icon(val_bin.exists())} val.bin exists")
    
    if dataset_ok:
        size_mb = sum(f.stat().st_size for f in [train_bin, test_bin, val_bin]) / (1024**2)
        print(f"  📊 Total size: {size_mb:.1f} MB")
    
    all_good &= dataset_ok
    
    # Check teacher configs
    print("\n🎓 TEACHER CONFIGS CHECK:")
    
    teachers = {
        "constant": {
            "path": "configs/agents/constant/LayerwiseNanoGPT/default.json",
            "expected_keys": ["agent_type", "initial_learning_rate"],
            "hpo_value": 0.000533703276260396
        },
        "exponential_decay": {
            "path": "configs/agents/exponential_decay/LayerwiseNanoGPT/default.json",
            "expected_keys": ["agent_type", "initial_learning_rate", "decay_rate", "decay_steps"],
            "hpo_value": 0.0017654048052495078
        },
        "step_decay": {
            "path": "configs/agents/step_decay/LayerwiseNanoGPT/default.json",
            "expected_keys": ["agent_type", "initial_learning_rate", "gamma", "step_size"],
            "hpo_value": 0.0014618962793704966
        },
        "sgdr": {
            "path": "configs/agents/sgdr/LayerwiseNanoGPT/default.json",
            "expected_keys": ["agent_type", "initial_learning_rate", "t_i", "t_mult", "batches_per_epoch"],
            "hpo_value": 0.001653693718282443
        }
    }
    
    for teacher_name, config_info in teachers.items():
        config_path = Path(config_info["path"])
        
        if not config_path.exists():
            print(f"  ❌ {teacher_name}: Config file not found!")
            all_good = False
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check all expected keys exist
        has_all_keys = all(key in config for key in config_info["expected_keys"])
        
        # Check if using HPO-optimized value
        using_hpo = abs(config.get("initial_learning_rate", 0) - config_info["hpo_value"]) < 1e-10
        
        status = check_icon(has_all_keys and using_hpo)
        print(f"  {status} {teacher_name}")
        
        if not has_all_keys:
            missing = [k for k in config_info["expected_keys"] if k not in config]
            print(f"      ⚠️  Missing keys: {missing}")
            all_good = False
        
        if not using_hpo:
            print(f"      ⚠️  Not using HPO value! Current: {config.get('initial_learning_rate', 'N/A')}")
            print(f"      ⚠️  Expected: {config_info['hpo_value']}")
            all_good = False
        else:
            print(f"      ✓ Using HPO-optimized learning rate: {config.get('initial_learning_rate')}")
    
    # Check environment config
    print("\n🌍 ENVIRONMENT CONFIG CHECK:")
    env_config_path = Path("hydra_conf/env/LayerwiseNanoGPT/Shakespeare.yaml")
    
    if env_config_path.exists():
        print(f"  ✅ Shakespeare.yaml exists")
        with open(env_config_path, 'r') as f:
            content = f.read()
            if "dataset_name: Shakespeare" in content:
                print(f"  ✅ Dataset name correctly set")
            else:
                print(f"  ❌ Dataset name not set correctly")
                all_good = False
    else:
        print(f"  ❌ Shakespeare.yaml not found!")
        all_good = False
    
    # Check if data directory needs to be created
    print("\n📁 DATA DIRECTORY CHECK:")
    data_dir = Path("data/nanoGPT_shakespeare")
    
    if data_dir.exists():
        print(f"  ℹ️  Data directory already exists")
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"  ℹ️  Found {len(subdirs)} existing data subdirectories:")
            for subdir in subdirs:
                files = list(subdir.glob("*.pkl"))
                print(f"      - {subdir.name}: {len(files)} pickle files")
    else:
        print(f"  ℹ️  Data directory will be created during data generation")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL CHECKS PASSED! You're ready to start data collection!")
        print("\n📋 Next steps:")
        print("1. Run test: python main.py env=LayerwiseNanoGPT/Shakespeare teacher=constant mode=data_gen results_dir=test_shakespeare/constant_test env.num_runs=1 env.num_epochs=1 seed=0 device=cuda")
        print("2. If test passes, start full data collection (see NEXT_STEPS_ACTION_PLAN.md)")
    else:
        print("⚠️  SOME CHECKS FAILED! Please fix the issues above before proceeding.")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())

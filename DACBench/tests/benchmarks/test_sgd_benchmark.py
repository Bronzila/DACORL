from __future__ import annotations

import json
import os
import unittest
import numpy as np

from dacbench.benchmarks import SGDBenchmark
from dacbench.envs import SGDEnv


class TestSGDBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SGDBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), SGDEnv)

    def test_setup(self):
        bench = SGDBenchmark()
        assert bench.config is not None

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = SGDBenchmark("test_conf.json")
        assert bench.config.dummy == 0
        os.remove("test_conf.json")

    def test_save_conf(self):
        bench = SGDBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = SGDBenchmark()
        bench.read_instance_set()
        assert len(bench.config.instance_set.keys()) == 100
        inst = bench.config.instance_set[0]
        bench2 = SGDBenchmark()
        env = bench2.get_environment()
        assert len(env.instance_set.keys()) == 100
        # [3] instance architecture constructor functionally identical but not comparable
        assert inst[0] == env.instance_set[0][0]
        assert inst[1] == env.instance_set[0][1]

    def test_benchmark_env(self):
        bench = SGDBenchmark()
        env = bench.get_benchmark()
        assert issubclass(type(env), SGDEnv)

    def test_from_to_json(self):
        bench = SGDBenchmark()
        restored_bench = SGDBenchmark.from_json(bench.to_json())
        assert bench.config.keys() == restored_bench.config.keys(), f"Configs should have same keys"
        for k in bench.config.keys():
            if k in ["reward_range", "observation_space_kwargs", "torch_hub_model"]:
                assert np.allclose(bench.config[k], restored_bench.config[k]), f"Config values should be equal, got: {bench.config[k]} != {restored_bench.config[k]}"
            elif k == "layer_specification":
                for layer in bench.config[k]:
                    assert layer in restored_bench.config[k], f"Layer {layer} should be in restored config"
            elif k == "optimizer_params":
                for kk in bench.config[k].keys():
                    assert np.allclose(bench.config[k][kk], restored_bench.config[k][kk]), f"Config values should be equal, got: {bench.config[k][kk]} != {restored_bench.config[k][kk]}"
            else:
                assert bench.config[k] == restored_bench.config[k], f"Config values should be equal, got: {bench.config[k]} != {restored_bench.config[k]}"

    def test_seeding(self):
        bench = SGDBenchmark()
        bench.config.seed = 123
        mems = []
        instances = []

        for _ in range(2):
            env = bench.get_environment()
            env.instance_index = 0
            instances.append(env.get_instance_set())

            state, _ = env.reset()

            terminated, truncated = False, False
            mem = []
            step = 0
            while not (terminated or truncated) and step < 5:
                action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                mem.append([state, [reward, int(truncated), action]])
                step += 1
            mems.append(mem)
        assert len(mems[0]) == len(mems[1])
        assert instances[0] == instances[1]

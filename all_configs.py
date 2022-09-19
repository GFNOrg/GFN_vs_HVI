from itertools import product

# seeds = [3]
# env_configs = ["easy"]
# batch_sizes = [16]
seeds = range(100, 105)
env_configs = ["hard", "easy"]
batch_sizes = [16, 64]
# This represent the sampling_mode, init_temperature, init_epsilon, and final_epsilon
sampling_modes = [
    ("on_policy", None, None, None),
    ("off_policy", 1.0, 1.0, 0.0),
    ("off_policy", 1.0, 1.0, 0.1),
    ("pure_off_policy", 1.0, 1.0, 0.0),
    ("pure_off_policy", 1.0, 1.0, 0.1),
]
modes_and_learnPBs_and_baselines = [
    ("tb", True, "None"),
    ("tb", False, "None"),
    ("modified_db", True, "None"),
    ("modified_db", False, "None"),
    ("forward_kl", True, "local"),
    ("forward_kl", True, "global"),
    ("forward_kl", False, None),
    ("reverse_kl", True, "local"),
    ("reverse_kl", True, "global"),
    ("reverse_kl", False, "local"),
    ("reverse_kl", False, "global"),
    ("rws", True, "None"),
    ("rws", False, "None"),
    ("reverse_rws", True, "local"),
    ("reverse_rws", True, "global"),
    ("reverse_rws", False, "local"),
    ("reverse_rws", False, "global"),
    ("symmetric_cycles", True, "local"),
    ("symmetric_cycles", True, "global"),
    ("symmetric_cycles", False, "local"),
    ("symmetric_cycles", False, "global"),
]
all_configs = list(
    product(
        seeds,
        env_configs,
        batch_sizes,
        sampling_modes,
        modes_and_learnPBs_and_baselines,
    )
)
all_configs_list_of_dicts = []
for config in all_configs:
    seed, env_config, batch_size, sampling_mode, mode_and_learnPB_and_baseline = config
    mode, learn_PB, baseline = mode_and_learnPB_and_baseline
    sampling_mode, init_temperature, init_epsilon, final_epsilon = sampling_mode
    all_configs_list_of_dicts.append(
        {
            "seed": seed,
            "batch_size": batch_size,
            "env": env_config,
            "sampling_mode": sampling_mode,
            "mode": mode,
            "learn_PB": learn_PB,
            "baseline": baseline,
            "init_temperature": init_temperature,
            "init_epsilon": init_epsilon,
            "final_epsilon": final_epsilon,
        }
    )
total_configs_1 = len(all_configs_list_of_dicts)

new_sampling_modes = [
    ("off_policy", 1.0, 1.0, 0.02),
    ("pure_off_policy", 1.0, 1.0, 0.02),
]
new_modes_and_learnPBs_and_baselines = [
    ("tb", True, "None"),
    ("tb", False, "None"),
    ("modified_db", True, "None"),
    ("modified_db", False, "None"),
    ("forward_kl", True, "global"),
    ("forward_kl", False, None),
    ("reverse_kl", True, "global"),
    ("reverse_kl", False, "global"),
    ("rws", True, "None"),
    ("rws", False, "None"),
    ("reverse_rws", True, "global"),
    ("reverse_rws", False, "global"),
    ("symmetric_cycles", True, "global"),
    ("symmetric_cycles", False, "global"),
]
new_all_configs = list(
    product(
        seeds,
        env_configs,
        batch_sizes,
        new_sampling_modes,
        new_modes_and_learnPBs_and_baselines,
    )
)
for config in new_all_configs:
    seed, env_config, batch_size, sampling_mode, mode_and_learnPB_and_baseline = config
    mode, learn_PB, baseline = mode_and_learnPB_and_baseline
    sampling_mode, init_temperature, init_epsilon, final_epsilon = sampling_mode
    all_configs_list_of_dicts.append(
        {
            "seed": seed,
            "batch_size": batch_size,
            "env": env_config,
            "sampling_mode": sampling_mode,
            "mode": mode,
            "learn_PB": learn_PB,
            "baseline": baseline,
            "init_temperature": init_temperature,
            "init_epsilon": init_epsilon,
            "final_epsilon": final_epsilon,
        }
    )

total_configs = len(all_configs_list_of_dicts)


if __name__ == "__main__":
    print(all_configs_list_of_dicts)
    print(f"Total number of configurations before first addition: {total_configs_1}")
    print(f"Total number of configurations: {total_configs}")

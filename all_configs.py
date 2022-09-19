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

total_configs = len(all_configs_list_of_dicts)


# def populate_configs_list(
#     configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# ):
#     for seed in seeds:
#         for ndim, height, R0 in env_configs:
#             for mode in modes:
#                 for batch_size in batch_sizes:
#                     for init_temperature in init_temperatures:
#                         for learn_PB in learn_PBs:
#                             for use_replay_buffer in use_replay_buffers:
#                                 configs_list.append(
#                                     {
#                                         "seed": seed,
#                                         "ndim": ndim,
#                                         "height": height,
#                                         "R0": R0,
#                                         "mode": mode,
#                                         "batch_size": batch_size,
#                                         "init_temperature": init_temperature,
#                                         "learn_PB": learn_PB,
#                                         "use_replay_buffer": use_replay_buffer,
#                                     }
#                                 )


# all_configs_list = []
# total_configs = len(all_configs_list)


# seeds = range(10, 15)
# env_configs = [(4, 8, 0.1), (2, 64, 0.01)]
# modes = ["tb"]
# batch_sizes = [16, 64, 256]
# init_temperatures = [1.0, 2.0]
# learn_PBs = [False, True]
# use_replay_buffers = [False]

# populate_configs_list(
#     all_configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# )

# env_configs = [(4, 8, 0.1), (2, 64, 0.001)]
# modes = ["modified_db"]

# populate_configs_list(
#     all_configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# )

# env_configs = [(2, 64, 0.001)]
# modes = ["symmetric_cycles", "tb"]

# populate_configs_list(
#     all_configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# )

# use_replay_buffers = [True]
# modes = ["tb", "modified_db"]

# populate_configs_list(
#     all_configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# )

# env_configs = [(2, 64, 0.001), (4, 8, 0.1)]
# use_replay_buffers = [True, False]
# modes = ["reverse_kl", "reverse_rws"]

# populate_configs_list(
#     all_configs_list,
#     seeds,
#     env_configs,
#     modes,
#     batch_sizes,
#     init_temperatures,
#     learn_PBs,
#     use_replay_buffers,
# )

# total_configs = len(all_configs_list)

if __name__ == "__main__":
    print(all_configs_list_of_dicts)
    print(f"Total number of configurations: {total_configs}")

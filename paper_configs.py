env = "very_hard"
PB = "learnable"
batch_sizes = [64]
early_stop = 0
exploration_scheduling = "cosine"
final_epsilon = 0.0
init_epsilon = 0.0
final_temeprature = 0.0
temperature_sf = True


modes = (["tb", "modified_db", "reverse_kl", "forward_kl", "rws", "reverse_rws"],)
sampling_modes = (["on_policy", "pure_off_policy"],)
baselines = ["None", "local", "global"]
lrs = []

# mode, sampling_mode, baseline, exploration_phase_ends_by, init_temperature, lr, lr_PB, lr_Z, lr_scheduling, schedule
configs = [
    ("tb", "on_policy", "None", -1, 0, 4e-3, 3e-4, 0.04, "plateau", 0.9),
    ("tb", "on_policy", "None", -1, 0, 4e-3, 3e-4, 0.04, "None", 1),
    ("tb", "on_policy", "None", -1, 0, 5e-3, 1e-3, 0.05, "plateau", 0.5),
    ("tb", "on_policy", "None", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.5),
    ("tb", "on_policy", "None", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.1),
    ("tb", "pure_off_policy", "None", -5, 1, 5e-3, 1e-3, 0.05, "plateau", 0.5),
    ("tb", "pure_off_policy", "None", -10, 1, 5e-3, 1e-3, 0.05, "plateau", 0.5),
    ("tb", "pure_off_policy", "None", -5, 1, 5e-3, 1e-3, 0.05, "plateau", 0.9),
    ("tb", "pure_off_policy", "None", -10, 1, 5e-3, 1e-3, 0.05, "plateau", 0.9),
    ("modified_db", "on_policy", "None", -1, 0, 3e-4, 2e-4, 0.1, "plateau", 0.5),
    ("modified_db", "on_policy", "None", -1, 0, 6e-3, 3e-3, 0.05, "plateau", 0.9),
    ("modified_db", "on_policy", "None", -1, 0, 4e-3, 3e-4, 0.04, "plateau", 0.9),
    ("modified_db", "pure_off_policy", "None", -5, 1, 6e-3, 3e-3, 0.05, "plateau", 0.9),
    (
        "modified_db",
        "pure_off_policy",
        "None",
        -10,
        1,
        6e-3,
        3e-3,
        0.05,
        "plateau",
        0.9,
    ),
    ("reverse_kl", "on_policy", "local", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.1),
    ("reverse_kl", "on_policy", "local", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.5),
    ("reverse_kl", "on_policy", "global", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.1),
    ("reverse_kl", "on_policy", "global", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.5),
    ("reverse_kl", "pure_off_policy", "local", -10, 1, 7e-3, 2e-3, 0.1, "plateau", 0.5),
    ("reverse_kl", "pure_off_policy", "local", -5, 1, 7e-3, 2e-3, 0.1, "plateau", 0.5),
    (
        "reverse_kl",
        "pure_off_policy",
        "global",
        -10,
        1,
        7e-3,
        2e-3,
        0.1,
        "plateau",
        0.5,
    ),
    ("reverse_kl", "pure_off_policy", "global", -5, 1, 7e-3, 2e-3, 0.1, "plateau", 0.5),
    ("forward_kl", "on_policy", "local", -1, 0, 3e-3, 2e-4, 0.008, "plateau", 0.5),
    ("forward_kl", "on_policy", "local", -1, 0, 3e-3, 1e-3, 0.02, "None", 1),
    ("forward_kl", "on_policy", "global", -1, 0, 3e-3, 2e-4, 0.008, "plateau", 0.5),
    ("forward_kl", "on_policy", "global", -1, 0, 3e-3, 1e-3, 0.02, "None", 1),
    ("forward_kl", "pure_off_policy", "local", -10, 1, 3e-3, 2e-3, 0.02, "None", 1),
    (
        "forward_kl",
        "pure_off_policy",
        "local",
        -10,
        1,
        3e-3,
        2e-3,
        0.02,
        "plateau",
        0.5,
    ),
    ("forward_kl", "pure_off_policy", "global", -10, 1, 3e-3, 2e-3, 0.02, "None", 1),
    (
        "forward_kl",
        "pure_off_policy",
        "global",
        -10,
        1,
        3e-3,
        2e-3,
        0.02,
        "plateau",
        0.5,
    ),
    ("rws", "on_policy", "None", -1, 0, 1e-3, 1e-3, 0.05, "plateau", 0.5),
    ("rws", "on_policy", "None", -1, 0, 7e-4, 3e-4, 0.02, "None", 1),
    ("rws", "pure_off_policy", "None", -10, 1, 7e-4, 3e-4, 0.02, "None", 1),
    ("rws", "pure_off_policy", "None", -5, 1, 7e-4, 3e-4, 0.02, "None", 1),
    ("rws", "pure_off_policy", "None", -10, 1, 1e-3, 1e-3, 0.05, "plateau", 0.5),
    ("rws", "pure_off_policy", "None", -5, 1, 7e-4, 3e-4, 0.02, "None", 1),
    ("reverse_rws", "on_policy", "local", -1, 0, 3e-3, 8e-4, 0.07, "plateau", 0.9),
    ("reverse_rws", "on_policy", "local", -1, 0, 3e-3, 1e-3, 0.02, "None", 1),
    ("reverse_rws", "on_policy", "global", -1, 0, 3e-3, 8e-4, 0.07, "plateau", 0.9),
    ("reverse_rws", "on_policy", "global", -1, 0, 3e-3, 1e-3, 0.02, "None", 1),
    ("reverse_rws", "pure_off_policy", "local", -3, 1, 3e-3, 1e-3, 0.02, "None", 1),
    (
        "reverse_rws",
        "pure_off_policy",
        "local",
        -10,
        1,
        3e-3,
        1e-3,
        0.02,
        "plateau",
        0.5,
    ),
    ("reverse_rws", "pure_off_policy", "global", -10, 1, 3e-3, 1e-3, 0.02, "None", 1),
    (
        "reverse_rws",
        "pure_off_policy",
        "global",
        -10,
        1,
        3e-3,
        1e-3,
        0.02,
        "plateau",
        0.5,
    ),
]

configs_dict = [
    {
        "env": "very_hard",
        "PB": "learnable",
        "temperature_sf": True,
        "early_stop": 0,
        "gradient_estimation_interval": 1000,
        "validation_interval": 200,
        "mode": mode,
        "sampling_mode": sampling_mode,
        "baseline": baseline,
        "exploration_phase_ends_by": exploration_phase_ends_by,
        "init_temperature": init_temperature,
        "lr": lr,
        "lr_PB": lr_PB,
        "lr_Z": lr_Z,
        "lr_scheduling": lr_scheduling,
        "schedule": schedule,
    }
    for mode, sampling_mode, baseline, exploration_phase_ends_by, init_temperature, lr, lr_PB, lr_Z, lr_scheduling, schedule in configs
]

all_configs_dict = []
for seed in range(100, 110):
    for replay_capacity in [1000, 10000]:
        for config_dict in configs_dict:
            if config_dict["sampling_mode"] == "on_policy" and replay_capacity != 1000:
                continue
            config_dict["seed"] = seed
            config_dict["replay_capacity"] = replay_capacity
            all_configs_dict.append(config_dict)

print(all_configs_dict)
print(len(all_configs_dict))

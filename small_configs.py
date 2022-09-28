# mode, sampling_mode, baseline, exploration_phase_ends_by, init_temperature, lr, lr_PB, lr_Z, lr_scheduling, schedule
configs = [
    ("tb", "on_policy", "None", -1, 0, 4e-3, 3e-4, 0.04, "plateau", 0.9),
    ("tb", "on_policy", "None", -1, 0, 1e-3, 1e-3, 0.1, "plateau", 0.9),
    ("reverse_kl", "on_policy", "global", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.9),
    ("reverse_kl", "on_policy", "None", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.9),
    ("reverse_kl", "on_policy", "local", -1, 0, 1e-3, 7e-3, 0.01, "plateau", 0.9),
    ("forward_kl", "on_policy", "local", -1, 0, 3e-3, 2e-4, 0.008, "plateau", 0.9),
    ("forward_kl", "on_policy", "global", -1, 0, 3e-3, 2e-4, 0.008, "plateau", 0.9),
    ("forward_kl", "on_policy", "None", -1, 0, 3e-3, 2e-4, 0.008, "plateau", 0.9),
    ("rws", "on_policy", "None", -1, 0, 1e-3, 1e-3, 0.05, "plateau", 0.9),
    ("reverse_rws", "on_policy", "global", -1, 0, 3e-3, 1e-3, 0.02, "plateau", 0.9),
    ("reverse_rws", "on_policy", "local", -1, 0, 3e-3, 1e-3, 0.02, "plateau", 0.9),
    ("reverse_rws", "on_policy", "None", -1, 0, 3e-3, 1e-3, 0.02, "plateau", 0.9),
]

configs_dict = [
    {
        "PB": "learnable",
        "temperature_sf": True,
        "early_stop": 0,
        "gradient_estimation_interval": 0,
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

small_configs_dict = []
for seed in range(105, 110):
    for env in ["easy", "medium2"]:
        for config_dict in configs_dict:
            config = config_dict.copy()
            config["seed"] = seed
            config["env"] = env
            small_configs_dict.append(config)


if __name__ == "__main__":
    print(len(small_configs_dict))

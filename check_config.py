def check_args(args):

    if args.init_epsilon < args.final_epsilon:
        raise ValueError("init_epsilon must be greater than final_epsilon")
    if args.init_temperature < args.final_temperature:
        raise ValueError("init_temperature must be greater than final_temperature")

    if args.baseline != "None":
        assert (args.mode, args.PB) in [
            ("forward_kl", "learnable"),
            ("forward_kl", "tied"),
        ] or args.mode in ["reverse_kl", "reverse_rws", "symmetric_cycles"]

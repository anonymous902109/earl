from src.earl.algorithms.star_gan.main import get_parser, main


def train_star_gan(image_size=176, image_channels=3, c_dim=5, batch_size=16, domains=None, agent=None,
                   lambda_counter=1., counter_mode="raw", agent_type="deepq", ablate_agent=False,
                   num_iters=10000, save_path='trained_models', dataset_path="./data"):
    """
    Trains StarGAN on the given data set.

    :param dataset: Data set name. The data set is assumed to be saved in res/datasets/.
    :param name: Name under which the StarGAN models are saved. This will create a directory in res/models.
    :param image_size: The size of images within the data set (quadratic images are assumed).
    :param image_channels: Amount of image channels.
    :param c_dim: Amount of domains.
    :param batch_size: Batch size.
    :param agent_file: Path to the agent that should be used for the counterfactual loss. If no agent is given, StarGAN
        will be trained without a counterfactual loss.
    :param agent_type: The type of agent. "deepq" for Keras Deep-Q, "acer" for gym baselines ACER and "olson" for a
        Pytorch Actor Critic Space Invaders model.
    :param lambda_counter: Weight for the counterfactual loss.
    :param counter_mode: Mode of the counterfactual loss. Supported modes are "raw", "softmax", "advantage" and
        "z-score".
    :param ablate_agent: Whether the laser canon should be removed from space invaders frames before they are input to
        the agent.
    :return: None
    """
    args = [
        "--mode=train",
        "--datasets=RaFD",
        f"--rafd_crop_size={image_size}",
        f"--image_size={image_size}",
        f"--image_channels={image_channels}",
        f"--c_dim={c_dim}",
        f"--domains={domains}",
        f"--batch_size={batch_size}",
        f"--rafd_image_dir=../res/datasets/train",
        f"--sample_dir=../res/models/samples",
        f"--log_dir=trained_models/logs",
        f"--model_save_dir={save_path}",
        f"--result_dir=../res/models/results",
        f"--lambda_counter={lambda_counter}",
        f"--counter_mode={counter_mode}",
        f"--agent_type={agent_type}",
        f"--ablate_agent={ablate_agent}",
        f"--dataset_path={dataset_path}",
        f"--num_iters={num_iters}",
        "--num_iters_decay=10000",
        "--log_step=100",
        "--sample_step=250000",
        f"--model_save_step={int(num_iters/10)}",
        "--use_tensorboard=False",
    ]

    parser = get_parser()
    config = parser.parse_args(args)
    main(config, agent)

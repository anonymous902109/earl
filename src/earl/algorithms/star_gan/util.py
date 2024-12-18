import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
import keras
from torch.autograd import Variable




def run_agent(max_steps, agent, env_name, seed=None, max_noop=1, render=True, power_pill_objective=False,
              max_episodes=None, agent_type="keras", ablate_agent=False):
    """
    Runs the given agent on the given environment. This is mainly used to measure the performance of agents and for
    debugging. The accumulated reward is printed after every episode and the mean and std reward over all episodes is
    printed at the end.

    :param max_steps: Maximum amount of total steps until termination.
    :param agent: A trained agent (Pytorch and Keras are supported).
    :param env_name: The Gym environment name.
    :param seed: A random number generation seed.
    :param max_noop: A maximum amount of NOOPs that are executed at the start of each episode.
    :param render: Whether to render frames.
    :param power_pill_objective: Whether the Power-Pill objective is used on Pac-Man.
    :param max_episodes: Maximum amount of episodes until termination.
    :param agent_type: the type of Pacman Agent, ignored with Space Invader. Accepts "keras" or "acer".
    :param ablate_agent: Whether the laser canon should be hidden from the frames that are input to the agent.
    :return: None
    """
    if seed is not None:
        np.random.seed(seed)

    wrapper, skip_frames = init_environment(env_name, power_pill_objective, agent_type=agent_type,
                                            ablate_agent=ablate_agent)
    stacked_frames = wrapper.reset(noop_max=max_noop)

    total_reward = 0
    reward_list = []
    nb_episodes = 0

    for i in range(max_steps):
        if i < 4:
            action = wrapper.env.action_space.sample()
        else:
            output = get_agent_prediction_from_stacked_frames(agent, stacked_frames)
            action = np.argmax(np.squeeze(output))

        stacked_frames, observations, reward, done, info = wrapper.step(action, skip_frames=skip_frames)
        total_reward += reward
        if done:
            print('total_reward', total_reward)
            reward_list.append(total_reward)
            total_reward = 0
            nb_episodes += 1
            if max_episodes is not None and nb_episodes >= max_episodes:
                break

        if render:
            wrapper.env.render()

    wrapper.env.close()

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)

    print("AVG Reward:", average_reward)
    print("STD Reward:", std_reward)



def get_action_names(env_name):
    """
    Returns all a list of the action names of the given Gym environment.

    :param env_name: The Gym environment name.
    :return: List of action names.
    """
    env = gym.make(env_name)
    return env.unwrapped.get_action_meanings()


def restrict_tf_memory():
    """
    Restricts the tensorflow memory usage to be dynamic. If not used, tensorflow will greedily take memory.

    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_agent_action(agent, frame, pacman=True, agent_type="deepq", ablate_agent=False):
    """
    Gets the action that an agent would choose on the given single frame under a greedy policy. The given frame is
    copied 3 times for the input to get a 4-image input for atari agents.

    :param agent: The trained agent (Pytorch and Keras are supported).
    :param frame: The input frame to the agent.
    :param pacman: Whether the target environment is Pac-Man or Space Invaders.
    :param agent_type: The type of agent. "deepq" for keras deepq, "acer" for baselines acer, "torch" for a pytorch acer-critic
    :return: Integer that encodes the chosen action.
    """
    if not isinstance(frame, Image.Image):
        raise NotImplementedError("get_agent_action is only implemented for Image frames (not numpy arrays or other)")
    if frame.size == (176, 176):
        frame = frame.crop((8, 1, 168, 174))

    agent_prediction = get_agent_prediction(agent, frame, pacman=pacman, agent_type=agent_type,
                                            ablate_agent=ablate_agent)
    return int(np.argmax(np.squeeze(agent_prediction)))



def get_agent_prediction_from_stacked_frames(agent, stacked_frames):
    """
    Gets the unprocessed agent output of the given agent on the given stacked frames.

    :param agent: The trained agent (Pytorch and Keras are supported).
    :param stacked_frames: A list or array of 4 frames.
    :return: An action distribution (A list of numeric output values for each action).
    """
    stacked_frames = np.expand_dims(stacked_frames, axis=0)

    if isinstance(agent, keras.models.Model):
        # Keras
        output = agent.predict(stacked_frames)
        if len(output) == 2:
            # in case a dueling net is used
            output = output[0]
    else:
        # Pytorch
        torch_state = torch.Tensor(stacked_frames)
        output = agent.pi(agent(torch_state)).detach().numpy()

    return output


def denorm(x):
    """
    Converts the range of input x from [-1, 1] to [0, 1].

    :param x: Torch variable in range [-1, 1]
    :return: Torch variable in range [0, 1]
    """
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def array_to_pil_format(array):
    """
    Converts an input array to PIL's image format (range [0, 255] and 8-bit integer encoding).

    :param array: Array representing an image.
    :return: If the input is not already 8-bit encoded, returns the input array times 255 in 8-bit encoding.
    """
    if array.dtype != np.uint8:
        return (array * 255).astype(np.uint8)
    else:
        return array


def load_olson_models(agent_file, encoder_file, generator_file, q_file, p_file, z_dim=16, wae_z_dim=128,
                      agent_latent=256, action_size=5, pac_man=True):
    """
    Loads all models of a trained version of the approach of Olson et al.

    :param agent_file: Path to the trained agent, that was used to train the explainability approach.
    :param encoder_file: Path to the encoder.
    :param generator_file: Path to the generator.
    :param q_file: Path to the encoder Q from the Wasserstein Autoencoder.
    :param p_file: Path to the decoder P from the Wasserstein Autoencoder.
    :param z_dim: Size of the latent space Z.
    :param wae_z_dim: Size of the latent space of the Wasserstein Autoencoder.
    :param agent_latent: Size of the agent's latent space.
    :param action_size: Amount of actions/domains that the approach was trained on.
    :param pac_man: Whether the target environment is Pac-Man or Space Invaders.
    :return: (agent, encoder, generator, Q, P)
    """
    # init models
    if pac_man:
        if agent_file.endswith(".h5"):
            agent = KerasAgent(agent_file, num_actions=action_size, latent_size=agent_latent)
        elif agent_file.endswith(".pt"):
            agent = ACER_Agent(num_actions=action_size, latent_size=agent_latent).cuda()
            agent.load_state_dict(torch.load(agent_file))
    else:
        agent = olson_model.Agent(action_size, 32).cuda()
        # The line above seems to indicate that the code assumes agent latent space 32 for space invader.
        # To be sure and since we need it for Olson Q and P, I set it here:
        agent_latent = 32
        agent.load_state_dict(torch.load(agent_file, map_location=lambda storage, loc: storage))
    encoder = olson_model.Encoder(z_dim).cuda()
    generator = olson_model.Generator(z_dim, action_size, pac_man=pac_man).cuda()
    Q = olson_model.Q_net(wae_z_dim, agent_latent=agent_latent).cuda()
    P = olson_model.P_net(wae_z_dim, agent_latent=agent_latent).cuda()

    # load saved weights
    encoder.load_state_dict(torch.load(encoder_file, map_location="cuda:0"))
    generator.load_state_dict(torch.load(generator_file, map_location="cuda:0"))
    Q.load_state_dict(torch.load(q_file, map_location="cuda:0"))
    P.load_state_dict(torch.load(p_file, map_location="cuda:0"))
    Q.eval()
    P.eval()

    return agent, encoder, generator, Q, P


def load_baselines_model(path, num_actions = False, num_env = 4):
    """
    Loads a model of the baselines repository.

    :param path: Path to the trained agent, that was trained by the openai baselines repository.
    :param num_actions: Used to define custom amount of actions. For example to remove ambiguous actions in Pacman.
        If False, the nomral amount is used.
    :param num_env: The number of envs to learn in parallel. Should be equivalent to batchsize during GAN training.
    """
    args = []
    args.append("--alg=acer")
    args.append("--env=MsPacmanNoFrameskip-v4")
    args.append("--num_timesteps=0")
    args.append("--load_path=" + path)
    args.append("--play")
    args.append(f"--num_env={num_env}")

    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if num_actions:
        env.action_space = gym.spaces.Discrete(num_actions)

    if args.save_video_interval != 0:
        raise NotImplementedError("Video saving is not implemented here. Use the origninal Baselines repository.")

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model


def generate_counterfactual(generator, image, target_domain, nb_domains, image_size=176):
    """
    Generates a counterfactual frame for the given image with StarGAN.

    :param generator: The StarGAN generator.
    :param image: The PIL image to generate a counterfactual for.
    :param target_domain: The integer encoded target action/domain.
    :param nb_domains: Number of possible actions/domains.
    :param image_size: The image size of the input image and the counterfactual (squared images are assumed).
    :return: (counterfactual, generation_time) - The counterfactual is a PIL image and the generation time is the pure
        time spent for the forward call (without pre- or postprocessing).
    """
    # define preprocessing
    transform = get_star_gan_transform(image_size, image_size, len(image.getbands()))

    # load and preprocess example image
    image = transform(image).cuda()
    image = image.unsqueeze(0)

    # convert target class to onehot
    onehot_target_class = np.zeros(nb_domains, dtype=int)
    onehot_target_class[target_domain] = 1
    onehot_target_class = torch.tensor([onehot_target_class]).cuda()

    # generate counterfactual
    start_time = time.time()
    counterfactual = generator(image, onehot_target_class)
    generation_time = time.time() - start_time

    # convert to PIL image
    counterfactual = denorm(counterfactual)
    counterfactual = counterfactual.detach().permute(0, 2, 3, 1).cpu().numpy()
    counterfactual = np.squeeze(counterfactual, axis=0)
    counterfactual = (counterfactual * 255).astype(np.uint8)
    counterfactual = Image.fromarray(counterfactual)

    return counterfactual, generation_time


def generate_olson_counterfactual(image, target_domain, agent, encoder, generator, Q, P, is_pacman, ablate_agent,
                                  max_iters=5000):
    """
    Generates a counterfactual frame for the given image with a trained approach of Olson et al.

    :param image: The PIL image to generate a counterfactual for.
    :param target_domain: The integer encoded target action/domain.
    :param agent: The agent that was used to train the explainability approach.
    :param encoder: The trained encoder.
    :param generator: The trained generator.
    :param Q: The trained encoder Q from the Wasserstein Autoencoder.
    :param P: The trained decoder P from the Wasserstein Autoencoder.
    :param is_pacman: Whether the target environment is Pac-Man or Space Invaders.
    :Param ablate_agent: Whether the laser canon should be removed from the frame that is passed to the agent.
    :param max_iters: Maximum amount of iterations for the gradient descent in the agents latent space via the
        Wasserstein Autoencoder.
    :return: (counterfactual, generation_time) - The counterfactual is a PIL image and the generation time is the pure
        time spent for the forward call (without pre- or postprocessing).
    """
    state_rgb, state_bw = prepro(np.array(image), pacman=is_pacman)
    state = Variable(torch.Tensor(np.expand_dims(state_rgb, axis=0)).permute(0, 3, 1, 2)).cuda()
    if ablate_agent:
        agent_state = Variable(torch.Tensor(np.expand_dims(ablate_screen(state_bw, "agent"), axis=0)).cuda())
    else:
        agent_state = Variable(torch.Tensor(np.expand_dims(state_bw, axis=0)).cuda())
    agent_state = torch.cat([agent_state, agent_state.clone(), agent_state.clone(), agent_state.clone()], dim=1)
    np.set_printoptions(precision=4)

    start_time = time.time()
    # get latent state representations
    z = encoder(state)
    z_a = agent(agent_state)
    z_n = Q(z_a)

    # generate the counterfactual image
    counterfactual_z_n = olson_tec.generate_counterfactual(z_n, target_domain, agent, P, MAX_ITERS=max_iters)
    p_cf = F.softmax(agent.pi(P(counterfactual_z_n)), dim=1)
    counterfactual = generator(z, p_cf)
    generation_time = time.time() - start_time

    counterfactual = Image.fromarray((counterfactual[0].permute(1, 2, 0).cpu().data.numpy() * 255).astype(np.uint8))

    return counterfactual, generation_time


if __name__ == "__main__":
    restrict_tf_memory()

    # Settings
    pacman = True
    nb_actions = 5
    env_name = "MsPacmanNoFrameskip-v4"
    img_size = 176
    agent_file = "../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"
    agent = keras.models.load_model(agent_file)
    ablate_agent = False

    # Load a StarGAN generator
    generator = Generator(c_dim=nb_actions, channels=3).cuda()
    generator.load_state_dict(torch.load("../res/models/PacMan_Ingame/models/200000-G.ckpt",
                                         map_location=lambda storage, loc: storage))

    # Load all relevant models that are necessary for the CF generation of Olson et al. via load_olson_models()
    olson_agent, olson_encoder, olson_generator, olson_Q, olson_P = load_olson_models(
        agent_file,
        "../res/models/PacMan_Ingame_Olson/enc39",
        "../res/models/PacMan_Ingame_Olson/gen39",
        "../res/models/PacMan_Ingame_Olson_wae/Q",
        "../res/models/PacMan_Ingame_Olson_wae/P",
        action_size=nb_actions,
        pac_man=pacman)

    # Load the original frame and specify the target action
    original_frame = Image.open("../res/HIGHLIGHTS_DIV/Summaries/PacMan_Ingame/3_19.png")
    target_action = 2  # Corresponds to "RIGHT" on Pac-Man

    # Generate a counterfactual with StarGAN
    star_gan_cf, star_gan_generation_time = generate_counterfactual(generator, original_frame, target_action,
                                                                    nb_actions, image_size=img_size)

    # Generate a counterfactual with Olson et al.
    olson_cf, olson_generation_time = generate_olson_counterfactual(original_frame, target_action, olson_agent,
                                                                    olson_encoder, olson_generator, olson_Q, olson_P,
                                                                    pacman, ablate_agent)

    # Save CF images
    star_gan_cf.save("StarGAN_CF.png")
    olson_cf.save("Olson_CF.png")

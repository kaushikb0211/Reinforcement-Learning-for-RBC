#!/usr/bin/env python
import sys
import dedalus.public as d3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rbc_env_limited_obs import DedalusRBC_Env
import argparse
import os
import logging


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def log_environment_setup(env_idx, total_envs):
    """Log information about the environment setup."""
    logging.info(f"Setting up environment {env_idx + 1}/{total_envs}...")


def main(argv):
    # Set up logging
    setup_logging()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RBC PPO Training")
    parser.add_argument("--n_envs", type=int, default=15, help="Number of environments")
    parser.add_argument("--total_timesteps", type=int, default=300000, help="Timesteps per run")
    parser.add_argument("--config", type=str, default="1", help="config for limited observation")

    # Add observation boundary arguments
    parser.add_argument("--x_start", type=float, default=0.0, help="Start of the x observation range")
    parser.add_argument("--x_end", type=float, default=1.0, help="End of the x observation range")
    parser.add_argument("--z_start", type=float, default=0.0, help="Start of the z observation range")
    parser.add_argument("--z_end", type=float, default=1.0, help="End of the z observation range")

    args = parser.parse_args(argv)

    # Log received arguments
    logging.info(f"Parsed arguments: n_envs={args.n_envs}, total_timesteps={args.total_timesteps}, config={args.config}")
    logging.info(f"Parsed boundaries: x_start={args.x_start}, x_end={args.x_end}, z_start={args.z_start}, z_end={args.z_end}")


    # Pass these arguments to the environment
    env_kwargs = {
        "x_start": args.x_start,
        "x_end": args.x_end,
        "z_start": args.z_start,
        "z_end": args.z_end
    }


    # Define the log and model directories

    run_name = f"config{args.config}_n_envs_{args.n_envs}_steps_{args.total_timesteps}"

    main_dir = "/scratch/f93919kb/lim_obs/"
    model_dir = os.path.join(main_dir, "models", run_name)
    log_dir = os.path.join(main_dir, "logs", run_name)

    # Define model save path
    model_name = f"ppo_rbc_config{args.config}.zip"

    # Create directories for logs and models
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, model_name)

    # Log basic setup information
    logging.info("====================================")
    logging.info("Initializing PPO Training")
    logging.info(f"Number of environments: {args.n_envs}")
    logging.info(f"Total timesteps: {args.total_timesteps}")
    logging.info(f"Log directory: {log_dir}")
    logging.info(f"Model directory: {model_dir}")
    logging.info(f"Limited observation configuration: {args.config}")
    logging.info("====================================")

    # Log environment creation
    logging.info("Creating vectorized environment...")
    for env_idx in range(args.n_envs):
        log_environment_setup(env_idx, args.n_envs)

    env = make_vec_env(DedalusRBC_Env, n_envs=args.n_envs, seed=0, env_kwargs=env_kwargs)

    # Check if a previous model exists and load it
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
    else:
        logging.info("No model found, creating a new one.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=256,  # Number of steps per environment per update, reduced due to memory issues
            batch_size=256,
            policy_kwargs=dict(net_arch=[512, 512]),
            learning_rate=1e-4,
            ent_coef=0.01
        )

    # Set up checkpoint callback
    logging.info("Setting up checkpoint callback...")
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Save model every 1000 steps - this is multiplied by the number of environments
        save_path=model_dir,
        name_prefix=f"ppo_rbc_",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Log training start
    logging.info("====================================")
    logging.info(f"Training model for {args.total_timesteps} timesteps with {args.n_envs} environments...")
    logging.info("====================================")

    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Log model saving
    logging.info(f"Training complete. Saving model to {checkpoint_path}")
    model.save(checkpoint_path)

    # Final log message
    logging.info("====================================")
    logging.info("Model training and saving complete.")
    logging.info("====================================")


if __name__ == "__main__":
    main(sys.argv[1:])

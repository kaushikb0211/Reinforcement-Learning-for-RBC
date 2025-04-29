#!/usr/bin/env python
import sys
import dedalus.public as d3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rbc_env import DedalusRBC_Env
import argparse
import os
import logging
from stable_baselines3.common.vec_env import VecNormalize


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
    parser.add_argument("--n_envs", type=int, default=5, help="Number of environments")
    parser.add_argument("--total_timesteps", type=int, default=500, help="Timesteps per run")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Base directory for logs")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save/load models")
    parser.add_argument("--job_name", type=str, default="default_job", help="Job name for dynamic checkpoint naming")
    parser.add_argument("--seed", type=int, default=0, help="change the value of random seed")
    args = parser.parse_args(argv)

    # Log basic setup information
    logging.info("====================================")
    logging.info("Initializing PPO Training")
    logging.info(f"Number of environments: {args.n_envs}")
    logging.info(f"Total timesteps: {args.total_timesteps}")
    logging.info(f"Log directory: {args.log_dir}")
    logging.info(f"Model directory: {args.model_dir}")
    logging.info("====================================")

    # Define a name for the model and log directories
    run_name = f"{args.job_name}_n_envs_{args.n_envs}_steps_{args.total_timesteps}"
    
    # Define the log and model directories
    tensorboard_log_path = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)

    # Ensure model and log directory exists
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)


    model_name = "ppo_rbc.zip"  
    checkpoint_path = os.path.join(model_dir, model_name)

    # Log where the model will be saved
    logging.info(f"Model will be saved to: {checkpoint_path}")


    # Log environment creation

    # Create a vectorized environment
    env = make_vec_env(DedalusRBC_Env, n_envs=args.n_envs, seed=args.seed)
    

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
            tensorboard_log=tensorboard_log_path,
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
        name_prefix=f"ppo_rbc_{args.job_name}",  #  this will also automatically include the number of timesteps
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Log training start
    logging.info("====================================")
    logging.info(f"Training model for {args.total_timesteps} timesteps with {args.n_envs} environments...")
    logging.info(f"Training using seed {args.seed}")
    logging.info("====================================")

    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save model
    logging.info(f"Saving model to {checkpoint_path}")
    model.save(checkpoint_path)

    logging.info("Training complete.")

    # Final log message
    logging.info("====================================")
    logging.info("Model training and saving complete.")
    logging.info("====================================")


if __name__ == "__main__":
    main(sys.argv[1:])
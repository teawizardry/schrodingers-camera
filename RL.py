# %%
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%
from helper import MetricLogger
import datetime
from pathlib import Path
from quantumEnv import QuantumEnv
from noise_machine.quantumEnv import NoiseQuantumEnv
from agent import Agent
import wandb
import argparse

# %%
# argparsing 
parser = argparse.ArgumentParser()
parser.add_argument("--gamma","-g", type=float, default=0.1, help="discount factor")
parser.add_argument("--learn_every", "-le", type=int, default=5, help="learning rate")
parser.add_argument("--sync_every","-se", type=int, default=30, help="target update rate")
parser.add_argument("--name", "-n", default="experiment", help="wandb experiment name")
parser.add_argument("--burnin", "-bi", type=int, default=1000, help="burn in")
parser.add_argument("--episodes", "-e", type=int, default=1000, help="episodes")
parser.add_argument("--checkpoint", "-chkpt", default=None, help="checkpoint path")
parser.add_argument("--memory_chkpt", "-mem_chkpt", default=None, help="memory checkpoint path")
parser.add_argument("--exploration_rate", "-er", type=float, default=1, help="exploration rate start")
parser.add_argument("--exploration_rate_decay", "-erd", type=float, default=0.99999975, help="exploration rate decay")
parser.add_argument("--noise_machine", "-nm", type=bool, default=False, help="pick normal operation or noise machine")
parser.add_argument("--batch_size", "-bs", type=int, default=32, help="batch size")
parser.add_argument("--adam", "-adam", type=bool, default=False, help="true-adam, false-sgd")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="agent learning rate")
parser.add_argument("--model", "-m", type=bool, default=False, help="False: normal agent, True: agent 2")
args = parser.parse_args()

# %%
# Training
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = args.checkpoint if args.checkpoint == None else Path(args.checkpoint)
mem_checkpoint = args.memory_chkpt if args.memory_chkpt == None else Path(args.memory_chkpt)

# logging setup
logger = MetricLogger(save_dir)
wandb.init(
      project="quantum-privacy", 
      name=f"{args.name}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}", 
      config={
        "batch_size": args.batch_size,
        "exploration_rate": args.exploration_rate,
        "exploration_rate_decay": args.exploration_rate_decay,
        "exploration_rate_min": 0.001,
        "gamma": args.gamma,
        "burnin": args.burnin,
        "learn_every": args.learn_every,
        "sync_every": args.sync_every,
        "save_every": 500,
        "episodes": args.episodes,
        "learning_rate": args.learning_rate,
        "model_selection": args.model
      })
config = wandb.config
# print(config)

# initialize environment
if args.noise_machine is False:
    env = QuantumEnv(render_mode=False)
else:
    env = NoiseQuantumEnv(render_mode=False)

# initialize agent
agent = Agent(state_dim=(args.batch_size, 3135), action_dim=env.action_space.n, save_dir=save_dir, config=config, checkpoint=checkpoint, memory_checkpoint=mem_checkpoint, adam=args.adam, noise_machine=args.noise_machine)

episodes = config.episodes

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # 1. Run agent on the state
        action = agent.act(state)
        print("action: ", action)
        wandb.log({"action": action})

        # 2. Agent performs action
        next_state, reward, done, info = env.step(action)
        # print(next_state)
        # print(next_state.shape)

        # 3. Remember
        agent.cache(state, next_state, action, reward, done)

        # 4. Learn
        q, q_target, loss = agent.learn()
        # print("q target: ", q_target)

        # 5. Logging
        logger.log_step(reward, loss, q, q_target) 

        # 6. Update state
        state = next_state

        # 7. Check if done
        if done:
            break
        
    logger.log_episode()
    agent.scheduler.step()

    if e % 1 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

wandb.finish()
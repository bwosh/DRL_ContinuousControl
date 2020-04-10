import numpy as np
import time

from tqdm import tqdm
from unityagents import UnityEnvironment

from ddpg.agent import Agent
from opts import Opts
from utils.plot import save_plot_results

# Parameters
opts = Opts()

# Create environment
env = UnityEnvironment(file_name='../Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Gather environment properties
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = states.shape[1]

agent = Agent(20, state_size, action_size, opts)

def play(brain_name, agent, env, pbar):
    # Reset environment and variables
    env_info = env.reset(train_mode=True)[brain_name]      
    states = env_info.vector_observations                  
    scores = np.zeros(num_agents)                          

    while True:
        # Act & get results
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name] 

        # Gather data
        rewards = env_info.rewards                        
        dones = env_info.local_done                        
        scores += env_info.rewards                        
        next_states = env_info.vector_observations 

        # Make agent step
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states 

        pbar.update()

        if np.any(dones):
            break
    return scores

# Try to solve environment
episode_scores = []
pbar = tqdm(total=opts.episodes*opts.moves_per_episode)
for episode in range(opts.episodes):
    # Display data
    pbar.set_description(f"E{episode+1}/{opts.episodes}")
    e_start = time.time()

    # Play
    scores = play(brain_name, agent, env, pbar)

    # Save scores
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    episode_scores.append(scores)

    # Solve rule
    mean_target_score = np.mean(episode_scores[-opts.target_score_episodes:])
    if len(episode_scores) >= opts.target_score_episodes and mean_target_score>=opts.target_avg_score:
        print(f"Environment solved after : {episode+1} episodes.")
        print(f"Mean score: {mean_target_score:.3f} over last {opts.target_score_episodes} episodes.")
        break

    agent.save()

    # Display
    e_stop = time.time()
    seconds = e_stop-e_start
    print(f"[Episode {episode}, Time(s): {seconds:.1f}] Score: {avg_score:.3f}, MeanOver{opts.target_score_episodes}: {mean_target_score:.3f}, Max={max_score:.3f}")

# Finish
env.close()

# Save training plot
save_plot_results(opts.approach_title, np.mean(episode_scores, axis=1), opts.target_score_episodes, opts.target_avg_score)
from unityagents import UnityEnvironment
import numpy as np

from agent import Agent
from plot import save_plot_results
from utils import StateAggregator

# Parameters
approach_title = "RANDOM"
episodes = 150
frames = 4
target_avg_score = 30
target_score_episodes = 100

# Create environment
env = UnityEnvironment(file_name='./Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations

num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = states.shape[1]

agent = Agent(state_size, action_size)

def play(brain_name, agent, env):
    # Reset environment and variables
    env_info = env.reset(train_mode=True)[brain_name]      
    states = env_info.vector_observations                  
    scores = np.zeros(num_agents)                          

    state_agg = StateAggregator(states, frames)
    next_state_agg = StateAggregator(states, frames)

    while True:
        # Choose actions
        actions = []
        for n_agent in range(num_agents):
            action = agent.act(states[n_agent])
            actions.append(action)
        actions = np.array(actions)
        actions = np.clip(actions, -1, 1)

        # Interact with env
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment

        # Gather data
        rewards = env_info.rewards                        
        dones = env_info.local_done                        
        scores += env_info.rewards                        
        next_states = env_info.vector_observations         # get next state (for each agent)

        next_state_agg.push(next_states)

        # Inform agent about the results
        aggregated_states = state_agg.to_input()
        aggregated_next_states = next_state_agg.to_input()
        for n_agent in range(num_agents):
            agent.step(aggregated_states[n_agent], actions[n_agent], rewards[n_agent], 
                        aggregated_next_states[n_agent], dones[n_agent])

        # Finish episode step
        states = next_states 
        state_agg.push(states)
        if np.any(dones):
            break
    return scores

# Try to solve environment
episode_scores = []
for episode in range(episodes):
    scores = play(brain_name, agent, env)
    avg_score = np.mean(scores)
    episode_scores.append(scores)

    # Solve rule
    mean_target_score = np.mean(episode_scores[-target_score_episodes:])
    if len(episode_scores) >= target_score_episodes and mean_target_score>=target_avg_score:
        print(f"Environment solved after : {episode+1} episodes.")
        print(f"Mean score: {mean_target_score:.3f} over last {target_score_episodes} episodes.")
        break

    print(f"[Episode {episode}] Score: {avg_score:.3f}, MeanOver{target_score_episodes}: {mean_target_score:.3f}")

# Finish
env.close()
save_plot_results(approach_title, np.mean(episode_scores, axis=1), target_score_episodes, target_avg_score)




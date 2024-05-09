# import subprocess
# import re

# # Define the command to run the ReflexAgent on the testClassic layout 5 times quietly
# command = ["python", "pacman.py", "-p", "ReflexAgent", "-l", "mediumClassic", "-n", "200", "-q"]

# # Run the command and capture the output
# result = subprocess.run(command, capture_output=True, text=True)

# # Output results
# output = result.stdout
# print("Full Output:\n", output)

# # Initialize variables
# individual_scores = re.findall(r"Score: (\d+)", output)
# average_score = None
# win_rate = None
# record = None

# # Safe extraction of average score
# average_score_match = re.search(r"Average Score: ([\d\.]+)", output)
# if average_score_match:
#     average_score = average_score_match.group(1)

# # Safe extraction of win rate, accounting for the new format
# win_rate_match = re.search(r"Win Rate:\s+\d+/\d+ \(([\d\.]+)\)", output)
# if win_rate_match:
#     win_rate = win_rate_match.group(1)

# # Safe extraction of game record
# record_match = re.search(r"Record: (.+)$", output, re.MULTILINE)
# if record_match:
#     record = record_match.group(1)

# # Display parsed results
# print("Individual Game Scores:", ", ".join(individual_scores))
# print("Average Score:", average_score if average_score else "Not available")
# print("Win Rate:", f"{win_rate}%" if win_rate else "Not available")
# print("Game Record:", record if record else "Not available")

#---------------------------------------------------------------------------------------------------------------------

# import subprocess
# import re

# # Define the agents and layouts to test
# # agents = ['ReflexAgent', 'MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']
# # layouts = ['testClassic', 'smallClassic', 'mediumClassic']

# agents = ['ReflexAgent', 'MinimaxAgent']
# layouts = ['testClassic']
# runs_per_setup = 10

# # Dictionary to store results
# results = {agent: {layout: {'average_score': [], 'win_rate': []} for layout in layouts} for agent in agents}

# # Execute games and collect data
# for agent in agents:
#     for layout in layouts:
#         print(f"\nTesting {agent} on {layout}")
#         total_scores = []
#         win_rates = []

#         for _ in range(runs_per_setup):
#             # command = ["python", "pacman.py", "-p", agent, "-l", layout, "-n", "1", "-q"]
#             command = ["python", "pacman.py", "-p", agent, "-l", layout, "-n", "1"]
#             result = subprocess.run(command, capture_output=True, text=True)
#             output = result.stdout

#             # Extract average score and win rate
#             score_match = re.search(r"Average Score: ([\d\.\-]+)", output)
#             win_rate_match = re.search(r"Win Rate:\s+\d+/\d+ \(([\d\.]+)\)", output)
            
#             if score_match:
#                 total_scores.append(float(score_match.group(1)))
#             if win_rate_match:
#                 win_rates.append(float(win_rate_match.group(1)))

#         # Calculate average of averages and win rates
#         if total_scores:
#             results[agent][layout]['average_score'] = sum(total_scores) / len(total_scores)
#         if win_rates:
#             results[agent][layout]['win_rate'] = 100*(sum(win_rates) / len(win_rates))

#         # Display individual performance results for each layout and agent
#         print(f"Avg Score for {agent} on {layout}: {results[agent][layout]['average_score']}")
#         print(f"Win Rate for {agent} on {layout}: {results[agent][layout]['win_rate']}%")

# Optionally, summarize or compare results further if needed

#---------------------------------------------------------------------------------------------------------------------------

# import subprocess
# import re
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the agents and layouts to test
# # agents = ['ReflexAgent', 'MinimaxAgent', 'AlphaBetaAgent']
# # layouts = ['testClassic', 'smallClassic', 'mediumClassic']

# agents = ['ReflexAgent', 'MinimaxAgent','AlphaBetaAgent']
# layouts = ['testClassic', 'smallClassic', 'mediumClassic']
# runs_per_setup = 50

# # Dictionary to store results
# results = {agent: {layout: {'average_score': [], 'win_rate': []} for layout in layouts} for agent in agents}

# # Execute games and collect data
# for agent in agents:
#     for layout in layouts:
#         print(f"\nTesting {agent} on {layout}")
#         total_scores = []
#         win_rates = []

#         for _ in range(runs_per_setup):
#             command = ["python", "pacman.py", "-p", agent, "-l", layout, "-n", "1", "-q"]
#             result = subprocess.run(command, capture_output=True, text=True)
#             output = result.stdout

#             # Extract average score and win rate
#             score_match = re.search(r"Average Score: ([\d\.\-]+)", output)
#             win_rate_match = re.search(r"Win Rate:\s+\d+/\d+ \(([\d\.]+)\)", output)
            
#             if score_match:
#                 total_scores.append(float(score_match.group(1)))
#             if win_rate_match:
#                 win_rates.append(float(win_rate_match.group(1)))

#         # Calculate average of averages and win rates
#         if total_scores:
#             results[agent][layout]['average_score'] = sum(total_scores) / len(total_scores)
#         if win_rates:
#             results[agent][layout]['win_rate'] = 100*(sum(win_rates) / len(win_rates))

#         # Display individual performance results for each layout and agent
#         print(f"Avg Score for {agent} on {layout}: {results[agent][layout]['average_score']}")
#         print(f"Win Rate for {agent} on {layout}: {results[agent][layout]['win_rate']}%")

# # Plotting the results for each agent
# for agent in agents:
#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle(f"Performance of {agent}")

#     # Scores
#     scores = [results[agent][layout]['average_score'] for layout in layouts]
#     axs[0].bar(layouts, scores, color='blue')
#     axs[0].set_title('Average Scores by Layout')
#     axs[0].set_ylabel('Average Score')
#     axs[0].set_xlabel('Layout')
#     axs[0].set_ylim([min(scores) - 10, max(scores) + 10])

#     # Win rates
#     win_rates = [results[agent][layout]['win_rate'] for layout in layouts]
#     axs[1].bar(layouts, win_rates, color='green')
#     axs[1].set_title('Win Rates by Layout')
#     axs[1].set_ylabel('Win Rate (%)')
#     axs[1].set_xlabel('Layout')
#     axs[1].set_ylim([0, 100])

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


# # Plotting the comparison of agents across layouts
# fig, axs = plt.subplots(2, 1, figsize=(12, 12))
# fig.suptitle('Comparison of Agents across Layouts')

# # Data for plots
# x = np.arange(len(layouts))  # the label locations
# width = 0.20  # the width of the bars

# for i, agent in enumerate(agents):
#     scores = [results[agent][layout]['average_score'] for layout in layouts]
#     win_rates = [results[agent][layout]['win_rate'] for layout in layouts]

#     # Average Scores plot
#     rects1 = axs[0].bar(x + i*width, scores, width, label=agent)

#     # Win Rates plot
#     rects2 = axs[1].bar(x + i*width, win_rates, width, label=agent)

# # Add some text for labels, title and custom x-axis tick labels, etc.
# axs[0].set_ylabel('Average Scores')
# axs[0].set_title('Average Scores by Agent and Layout')
# axs[0].set_xticks(x + width)
# axs[0].set_xticklabels(layouts)
# axs[0].legend()

# axs[1].set_ylabel('Win Rate (%)')
# axs[1].set_title('Win Rates by Agent and Layout')
# axs[1].set_xticks(x + width)
# axs[1].set_xticklabels(layouts)
# axs[1].legend()

# plt.show()


#-------------------------------------------------------------------------------

import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import time

# Define the agents and layouts to test
agents = ['ReflexAgent', 'MinimaxAgent', 'AlphaBetaAgent', 'MonteCarloTreeSearchAgent']
layouts = ['testClassic', 'smallClassic', 'mediumClassic']
runs_per_setup = 10

# Dictionary to store results
results = {agent: {layout: {'average_score': [], 'win_rate': [], 'average_time': []} for layout in layouts} for agent in agents}

# Execute games and collect data
for agent in agents:
    for layout in layouts:
        print(f"\nTesting {agent} on {layout}")
        total_scores = []
        win_rates = []
        execution_times = []

        for run_number in range(runs_per_setup):
            print(f"Starting run {run_number + 1}/{runs_per_setup} for {agent} on {layout}")
            start_time = time.time()
            command = ["python", "pacman.py", "-p", agent, "-l", layout, "-n", "1", "-q"]
            result = subprocess.run(command, capture_output=True, text=True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            execution_times.append(elapsed_time)

            output = result.stdout

            # Extract average score and win rate
            score_match = re.search(r"Average Score: ([\d\.\-]+)", output)
            win_rate_match = re.search(r"Win Rate:\s+\d+/\d+ \(([\d\.]+)\)", output)
            
            if score_match:
                total_scores.append(float(score_match.group(1)))
            if win_rate_match:
                win_rates.append(float(win_rate_match.group(1)))

        # # Calculate average of averages, win rates, and times
        # results[agent][layout]['average_score'] = sum(total_scores) / len(total_scores)
        # results[agent][layout]['win_rate'] = 100*(sum(win_rates) / len(win_rates))
        # results[agent][layout]['average_time'] = sum(execution_times) / len(execution_times)
        
        # Calculate average of averages, win rates, and times
        average_score = sum(total_scores) / len(total_scores)
        average_win_rate = sum(win_rates) / len(win_rates)
        average_time = sum(execution_times) / len(execution_times)
        results[agent][layout]['average_score'] = average_score
        results[agent][layout]['win_rate'] = average_win_rate
        results[agent][layout]['average_time'] = average_time

        # Print summary for each layout and agent
        print(f"Summary for {agent} on {layout}:")
        print(f"Avg Score: {average_score:.2f}")
        print(f"Win Rate: {average_win_rate:.2f}%")
        print(f"Avg Time: {average_time:.2f}s")
        
        

# Summary of results
print("\nOverall performace across all runs and all layouts")
for agent in agents:
    avg_scores = np.mean([results[agent][layout]['average_score'] for layout in layouts])
    avg_win_rates = np.mean([results[agent][layout]['win_rate'] for layout in layouts])
    avg_times = np.mean([results[agent][layout]['average_time'] for layout in layouts])
    print(f"{agent}: Avg Score = {avg_scores:.2f}, Avg Win Rate = {avg_win_rates:.2f}%, Avg Time = {avg_times:.2f}s")


        
# Plotting the results for each agent
for agent in agents:
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(f"Performance of {agent}")

    # Scores
    scores = [results[agent][layout]['average_score'] for layout in layouts]
    axs[0].bar(layouts, scores, color='blue')
    axs[0].set_title('Average Scores by Layout')
    axs[0].set_ylabel('Average Score')
    axs[0].set_xlabel('Layout')
    axs[0].set_ylim([min(scores) - 10, max(scores) + 10])

    # Win rates
    win_rates = [results[agent][layout]['win_rate'] for layout in layouts]
    axs[1].bar(layouts, win_rates, color='green')
    axs[1].set_title('Win Rates by Layout')
    axs[1].set_ylabel('Win Rate (%)')
    axs[1].set_xlabel('Layout')
    # axs[1].set_ylim([0, 100])
    
    # Win rates
    times = [results[agent][layout]['average_time'] for layout in layouts]
    axs[2].bar(layouts, times, color='green')
    axs[2].set_title('Execution time by Layout')
    axs[2].set_ylabel('Avg Execution time (S)')
    axs[2].set_xlabel('Layout')
    # axs[2].set_ylim([0, 100])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
        
        
# Plotting the comparison of agents across layouts including execution times
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle('Comparison of Agents across Layouts')

# Data for plots
x = np.arange(len(layouts))  # the label locations
width = 0.20  # the width of the bars

for i, agent in enumerate(agents):
    scores = [results[agent][layout]['average_score'] for layout in layouts]
    win_rates = [results[agent][layout]['win_rate'] for layout in layouts]
    times = [results[agent][layout]['average_time'] for layout in layouts]

    # Average Scores plot
    axs[0].bar(x + i*width, scores, width, label=agent)
    # Win Rates plot
    axs[1].bar(x + i*width, win_rates, width, label=agent)
    # Execution Times plot
    axs[2].bar(x + i*width, times, width, label=agent)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs[0].set_ylabel('Average Scores')
axs[0].set_title('Average Scores by Agent and Layout')
axs[0].set_xticks(x + width)
axs[0].set_xticklabels(layouts)
axs[0].legend()

axs[1].set_ylabel('Win Rate (%)')
axs[1].set_title('Win Rates by Agent and Layout')
axs[1].set_xticks(x + width)
axs[1].set_xticklabels(layouts)
axs[1].legend()

axs[2].set_ylabel('Execution Time (s)')
axs[2].set_title('Average Execution Time by Agent and Layout')
axs[2].set_xticks(x + width)
axs[2].set_xticklabels(layouts)
axs[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
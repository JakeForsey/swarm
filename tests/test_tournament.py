from swarm import tournament
from swarm.agents import get_agent

def test_tournament():
    agent1 = "random"
    agent2 = "chaser"

    agents = [get_agent(agent1)]
    opponents = [get_agent(agent2)]
    
    results = tournament.run(
        agents=agents,
        opponents=opponents,
        num_rounds_per_matchup=1,
        episode_length=1
    )
    actual_result_names = {result["name"] for result in results}
    expected_result_names = {agent1, agent2}
    assert actual_result_names == expected_result_names

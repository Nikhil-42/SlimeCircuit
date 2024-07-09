# An Attempt to Perform 2-layer PCB Routing with Slime Mold Algorithm

The slime mold algorithm relies on agents which traverse the space and leave a trail of pheromones. The pheromones are used to attract other agents to the path. The algorithm is inspired by the behavior of slime molds in nature. The slime mold algorithm is used to solve optimization problems, such as the shortest path problem. In this project, the slime mold algorithm is used to perform 2-layer PCB routing.

## Basic Idea
- Each Net is represented by a unique specices of slime mold.
- Slime mold agents are placed on each of the nodes in the net.
- Agents are attracted to phereomones from their own species from different nodes.
- Agents are repelled by pheromones from other species.
- Agents ignore pheromones from their own node.
- Agents move along the path with the highest pheromone concentration.
- Agents leave pheromones on the path they traverse.
- When an agent reaches a different node, it becomes a new agent of the same species with the new node.
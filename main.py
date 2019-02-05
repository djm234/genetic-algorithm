from matplotlib import pyplot as plt

from itertools import combinations
import numpy as np
import copy

class Agent():
    def __init__(self, startx=0, starty=0, objective=[1,1], mutation_rate=0.02, n_steps=100):
        x, y = startx, starty
        self.pos = np.array([float(x), float(y)])
        self.start_pos = np.array([float(x), float(y)])
        self.xlim = [0, lx]
        self.ylim = [0, ly]
        self.vel = 1
        self.acc = 1
        self.mutation_rate = mutation_rate
        self.alive = True
        self.objective = objective
        self.n_steps = n_steps
        self.distance_travelled = 0.
        self.target_proximity = 0
        # Instructions are dx, dy, dvel, dacc
        self.instructions = [
            {
                'dpos': np.random.uniform([-1, 1], 2),
                'dacc': 0,
                'dvel': 0,
                'step': step,
            }
            for step in range(self.n_steps)
        ]
        self.calculate_fitness()
        return

    def move(self, instruction):
        self.check_if_alive()
        if self.alive:
            self.pos += instruction['dpos']
            self.vel = instruction['dvel']
            self.vel = instruction['dacc']
        else:
            # Dead, don't move
            self.pos += 0
            self.vel = 0
            self.vel = 0
        return

    def check_if_alive(self):
        self._hit_edge_of_box()
        self._hit_random_area_in_middle()
        return

    def _hit_random_area_in_middle(self):
        x, y = self.pos[0], self.pos[1]
        # rectangle:
        x1, x2 = 25, 75
        y1, y2 = 25, 75
        if (x1 < x < x2) and (y1 < y < y2):
            self.alive = False
        return

    def calculate_fitness(self):
        self.check_if_alive()
        self.check_current_proximity()
        #self.check_distance_travelled()
        self.fitness = self.target_proximity #+ self.distance_travelled
        return

    def _hit_edge_of_box(self):
        x, y = self.pos[0], self.pos[1]
        xmin, xmax = self.xlim[0], self.xlim[1]
        ymin, ymax = self.ylim[0], self.ylim[1]
        if (x < xmin) or (x > xmax):
            self.alive = False
        elif (y < ymin) or (y > ymax):
            self.alive = False
        return

    def _calc_distance(self, xy_a, xy_b):
        d = np.sqrt((xy_a[0] - xy_b[0])**2 + (xy_a[1] - xy_b[1])**2)
        return d

    def check_current_proximity(self):
        self.target_proximity = self._calc_distance(xy_a=self.pos, xy_b=self.objective)
        return

    def check_distance_travelled(self):
        distance = 0.
        xy_prev = self.start_pos
        for i, instruction in enumerate(self.instructions):
            distance += self._calc_distance(xy_prev, xy_prev+instruction['dpos'])
            xy_prev += xy_prev+instruction['dpos']
        self.distance_travelled = distance
        return

    def mutate(self):
        # In each case, test for change to mutate then mutate
        if np.random.uniform(0,1) <= self.mutation_rate:
            # randomly select between 1 and 10% of the indexes index
            i = np.random.choice(range(self.n_steps))
            instructions = self.instructions
            # Randomly change dpos
            instructions[i]['dpos'] = np.random.uniform([-1, 1], np.random.randint(0,int(0.1*len(instructions))))
            # Send new instructions back to the agent
            self.instructions = instructions
        elif np.random.uniform(0,1) <= self.mutation_rate:
            # Shuffle all of the dna
            np.random.shuffle(self.instructions)
        elif np.random.uniform(0,1) <= self.mutation_rate:
            # Shuffle a random portion of the dna
            instructions = self.instructions
            n = len(instructions)
            selection = {
                    1: [0, int(0.25*n)],
                    2: [int(0.25*n), int(0.50*n)],
                    3: [int(0.50*n), int(0.75*n)],
                    4: [int(0.75*n), n],
                    5: [int(0.9*n), n],
                    6: [0, int(0.1*n)],
                    6: [int(0.45*n), int(0.55*n)],
            }[np.random.choice([1, 2, 3, 4])]
            instructions_subset = instructions[selection[0]:selection[1]]
            np.random.shuffle(instructions_subset)
            instructions[selection[0]:selection[1]] = instructions_subset
        return

def reset_agents(all_agents, startx, starty, objective):
    for a in all_agents:
        a.alive = True
        a.pos = np.array([float(startx), float(starty)])
        a.objective = objective
    return all_agents

def crossover_dna(breeders):
    crossovers = []
    # For all possible pairs
    for pair in combinations(breeders, 2):
        # There is a chance they can swap half and half
        random_number = np.random.uniform()
        """if 0.0 <= random_number <= 0.25:
            a, b = pair
            assert len(a.instructions) == len(b.instructions)
            assert (type(a.instructions) is list) and (type(b.instructions) is list)
            n_instructions = len(a.instructions)
            # Create a child whose first half of instructions come from a, second half from b
            child1 = Agent()
            child1.instructions = a.instructions[:n_instructions//2] + b.instructions[n_instructions//2:]
            crossovers.append(child1)
            # And the opposite for child2
            child2 = Agent()
            child2.instructions = b.instructions[:n_instructions//2] + a.instructions[n_instructions//2:]
            crossovers.append(child2)
        """
        # A chance that random genes from A are taken from B
        if 0. <= random_number <= 0.5:
            a, b = pair
            assert len(a.instructions) == len(b.instructions)
            assert (type(a.instructions) is list) and (type(b.instructions) is list)
            n_instructions = len(a.instructions)
            # Create a child whose first half of instructions come from a, second half from b
            chosen_indices = np.random.choice(list(range(n_instructions)), n_instructions//2, replace=False)
            temp_instructions = np.array(a.instructions)
            temp_instructions[chosen_indices] = np.array(b.instructions)[chosen_indices]
            child1 = Agent()
            child1.instructions = list(temp_instructions)
            crossovers.append(child1)
            # And the opposite for child2
            child2 = Agent()
            temp_instructions = np.array(b.instructions)
            temp_instructions[chosen_indices] = np.array(a.instructions)[chosen_indices]
            child2 = Agent()
            child2.instructions = list(temp_instructions)
            crossovers.append(child2)
    return crossovers


if __name__ == '__main__':

    from IPython import embed

    n_agents = 100
    n_generations = 25
    n_breeders = 25  # gets big due to combinatorial epxplostion of mating
    lx, ly = 100, 100
    startx, starty = 0, 0
    objx, objy = 100, 100
    objective = np.array([objx, objy])

    all_agents = [
        Agent(startx, starty, objective)
        for i in range(n_agents)
    ]

    hall_of_fame_record_distances = [10000]
    hall_of_fame = []

    for generation in range(n_generations):
        prev_sum = sum([
            agent.fitness
            for agent in all_agents
        ])

        if generation > 0:
            # Select the best ones to keep
            all_agents = sorted(all_agents, key=lambda x: x.fitness, reverse=False)
            breeders = copy.deepcopy(all_agents[:n_breeders])
            crossovers = copy.deepcopy(crossover_dna(breeders))
            print(n_agents, len(breeders), len(crossovers), len(hall_of_fame))
            lucky_stragglers = copy.deepcopy(list(np.random.choice(all_agents, n_agents-(len(breeders)+len(crossovers)+len(hall_of_fame)), replace=False)))
            all_agents = breeders + hall_of_fame + crossovers + lucky_stragglers
            print("Keeping {} best from previous generation, {} HOF, {} crossovers, and {} stragglers (total={})".format(len(breeders), len(hall_of_fame), len(crossovers), len(lucky_stragglers), len(all_agents)))
            # After breeding, there is a chance the DNA can randomly change
            for agent in all_agents:
                agent.mutate()

        # Loop over each agent, run the instructions, and find target_proximity to target in the end
        all_agents = reset_agents(all_agents, startx, starty, objective)
        for agent in all_agents:
            # Carry out the moves
            for instruction in agent.instructions:
                agent.move(instruction)
            agent.calculate_fitness()

        print("\nGeneration {} top breeder start proximities:".format(generation))
        # Sort agents from smallest target_proximity to largest
        best = sorted(all_agents, key=lambda x: x.fitness, reverse=False)[:5]
        for agent in best:
            print(agent.target_proximity, agent.pos, agent.fitness)

        if (best[0].target_proximity < min(hall_of_fame_record_distances)):
            hall_of_fame.append(copy.deepcopy(best[0]))
            hall_of_fame_record_distances.append(best[0].target_proximity)
            print(hall_of_fame_record_distances)

        new_sum = sum([
            agent.fitness
            for agent in all_agents
        ])
        delta = prev_sum - new_sum
        if np.isnan(delta):
            embed()
        print("Delta sum:",prev_sum - new_sum)

print(hall_of_fame_record_distances)



fig = plt.figure()
import matplotlib.patches as patches
ax = fig.add_subplot(111, aspect='equal')
ax.add_patch(
     patches.Rectangle(
        (25, 25),
        width=50,
        height=50,
        color='k',
        fill=True
    )
)
ax.add_patch(
    patches.Rectangle(
        (0, 0),
        width=100,
        height=100,
        color='k',
        fill=False
    )
)

hall_of_fame = reset_agents(hall_of_fame, startx, starty, objective)
for agent in hall_of_fame:
    xs, ys = [], []
    for instruction in agent.instructions:
        agent.move(instruction)
        xs.append(agent.pos[0])
        ys.append(agent.pos[1])
    agent.calculate_fitness()
    fitness = agent.fitness
    ax.plot(xs, ys, '-o', label=fitness)
plt.axis('equal')
plt.xlim(xmin=0, xmax=100)
plt.ylim(ymin=0, ymax=100)
plt.savefig('offspring.png')
plt.show()

embed()

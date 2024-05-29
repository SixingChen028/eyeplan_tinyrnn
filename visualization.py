import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Visualizer:
    """
    A visualizer.
    """

    def __init__(self):
        """
        Construct an environment.
        """
    

    def reset_graph(self, child_dict):
        """
        Reset the graph.
        """

        self.child_dict = child_dict

        # create a directed graph
        self.G = nx.DiGraph()
        for parent, children in self.child_dict.items():
            for child in children:
                self.G.add_edge(parent, child)

        # get nodes
        self.nodes = list(self.G.nodes())
        self.root_node = list(self.child_dict.keys())[0]
        self.max_depth = len(self.nodes) // 2
    
    
    def get_pos(self):
        """
        Get positions of nodes.
        """
        # Circular layout for node positions
        pos = nx.circular_layout(self.G)
        pos = {k: pos[i] for i, k in enumerate(pos)}
        
        return pos
    

    def get_labels(self, rewards):
        """
        Get labels of nodes.
        """
        # Add node labels
        labels = {}
        for node in self.nodes:
            if node != self.root_node: # root node has no label
                labels[node] = rewards[node]
        
        return labels
    

    def plot(self, child_dict, rewards, arrows = None):
        """
        Plot the figure.
        """

        # reset graph
        self.reset_graph(child_dict)

        # get positions and labels
        pos = self.get_pos()
        labels = self.get_labels(rewards)

        # define node colors
        node_colors = []
        for node in self.nodes:
            if node == self.root_node:
                node_colors.append('skyblue')  # root node color
            else:
                node_colors.append('white')  # other nodes color

        # plot graph
        plt.figure(figsize = (4, 4))
        nx.draw(
            G = self.G,
            pos = pos,
            labels = labels,
            with_labels = True,
            node_size = 1000,
            node_color = node_colors,
            edgecolors = 'black',
            font_size = 12,
            arrowsize = 10,
        )

        # plot arrows
        if arrows != None:
            # Initialize colors
            start_color = np.array([1.0, 0.2, 0.3])  # Desaturated Orange
            end_color = np.array([1.0, 0.7, 0.0])    # Desaturated Red
            colors = np.linspace(start_color, end_color, len(arrows))

            pointer = 0
            for start, end in arrows:
                plt.annotate(
                    '',
                    xy = pos[end],
                    xytext = pos[start],
                    arrowprops = dict(
                        arrowstyle = 'fancy,head_length=0.6,head_width=0.6,tail_width=0.15',
                        connectionstyle = 'arc3,rad=0.2',
                        color = colors[pointer]
                    )
                )
                pointer += 1

        plt.show()


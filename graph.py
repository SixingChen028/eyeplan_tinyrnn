import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    """
    A graph class.
    """

    def __init__(self, num_node, reward_set):
        """
        Initialize the graph.
        """
        
        # initialize states
        self.num_node = num_node
        self.reward_set = reward_set
        

    def reset(self, shuffle_nodes = True, shuffle_edges = True):
        """
        Reset the graph.
        """

        # initialize and shuffle nodes
        self.nodes = np.arange(self.num_node)
        if shuffle_nodes:
            self.nodes = np.random.permutation(self.nodes)
        
        # initialize child node dict
        self.child_dict = {}
        for node_idx in np.arange(self.num_node, step = 2)[:-1]:
            if shuffle_edges:
                if node_idx == 0:
                    parent_idx = node_idx
                elif node_idx > 0:
                    parent_idx = np.random.choice([node_idx, node_idx - 1])
            else:
                parent_idx = node_idx

            child_idx = [node_idx + 1, node_idx + 2]

            self.child_dict[self.nodes[parent_idx]] = [self.nodes[child_idx[0]], self.nodes[child_idx[1]]]

        # initialize parent node dict
        self.parent_dict = {v: k for k, values in self.child_dict.items() for v in values}

        # get root node, leaf and non-leaf nodes
        self.root_node = self.nodes[0]
        self.non_leaf_nodes = np.array(list(self.child_dict.keys()))
        self.leaf_nodes = np.delete(self.nodes, np.where(np.isin(self.nodes, self.non_leaf_nodes))[0])
        
        # get node counts
        self.num_leaf = len(self.leaf_nodes)
        self.num_non_leaf = len(self.non_leaf_nodes)

        # initialize rewards
        self.rewards = np.random.choice(self.reward_set, size = self.num_node, replace = True)
        self.rewards[self.root_node] = 0.

        # get gains
        self.gains = np.zeros((self.num_node,))
        for node in self.nodes:
            self.gains[node] = self.compute_gain(node)


    def successors(self, node):
        """
        Find successor states of a given node.
        """

        if node in self.leaf_nodes:
            return [None, None]
        else:
            return self.child_dict[node]


    def predecessors(self, node):
        """
        Find predecessor states of a given node.
        """

        if node == self.root_node:
            return None
        else:
            return self.parent_dict[node]
    

    def transition(self, start_node, move):
        """
        World model giving prediction: (s, a) -> (s', r)
        """

        # no change if starting from a leaf node
        if start_node in self.leaf_nodes:
            end_node = start_node
            reward = 0.
        else:
            end_node = self.successors(start_node)[move]
            reward = self.rewards[end_node]

        return end_node, reward
    

    def compute_gain(self, node):
        """
        Compute cumulative reward to a node.
        """

        gain = self.rewards[node]

        # iteratively compute gains
        while node in self.parent_dict:
            parent_node = self.parent_dict[node]
            gain += self.rewards[parent_node]
            node = parent_node
        
        return gain
    

    def get_adj_list(self):
        """
        Get adjacency list.
        """

        self.adj_list = [[] for _ in range(self.num_node)]
        for node, children in self.child_dict.items():
            self.adj_list[node] = children

        return self.adj_list


    def get_adj_matrix(self):
        """
        Get adjacency matrix.
        """

        self.adj_matrix = np.zeros((self.num_node, self.num_node))
        for node, children in self.child_dict.items():
            for child in children:
                self.adj_matrix[node, child] = 1
        
        return self.adj_matrix


if __name__ == '__main__':
    # testing

    graph = Graph(num_node = 11, reward_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8]))
    graph.reset()

    print('child dict:', graph.child_dict)
    print('parent dict:', graph.parent_dict)
    print('rewards:', graph.rewards)
    print('gains:', graph.gains)
    print('a transition:', graph.transition(graph.root_node, 0))
    print('adjacency list:', graph.get_adj_list())
    print('adjacency matrix:', graph.get_adj_matrix())

    G = nx.from_numpy_array(graph.adj_matrix)
    plt.figure(figsize = (4, 4))
    nx.draw_circular(G, with_labels = True)
    plt.show()
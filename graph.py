import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx


def get_nx_graph(graph):
    mat=graph.get_adj_matrix()
    return nx.from_numpy_array(mat,create_using=nx.DiGraph())

    
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
        

    def reset(self, shuffle_nodes = True):
        """
        Reset the graph.
        """

        # initialize and shuffle nodes
        self.nodes = np.arange(self.num_node)
        if shuffle_nodes:
            self.nodes = np.random.permutation(self.nodes)
        
        # initialize leaf nodes and child node dict
        self.leaf_nodes = np.array([self.nodes[0]])
        self.child_dict = {}
        for idx in np.arange(1, self.num_node, step = 2):
            children = self.nodes[idx : idx + 2] # pick children
            parent = random.choice(self.leaf_nodes) # randomly pick a parent
            self.child_dict[parent] = children.tolist() # set children

            # update leaf nodes
            self.leaf_nodes = np.delete(self.leaf_nodes, np.where(np.isin(self.leaf_nodes, parent))[0]) # remove parent from leaves
            self.leaf_nodes = np.append(self.leaf_nodes, children) # add children to leaves

        # initialize parent node dict
        self.parent_dict = {v: k for k, values in self.child_dict.items() for v in values}

        # get root node and non-leaf nodes
        self.root_node = self.nodes[0]
        self.non_leaf_nodes = np.array(list(self.child_dict.keys()))

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
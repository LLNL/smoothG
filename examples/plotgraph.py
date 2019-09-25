from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

def read_graph(graph_file):
    raw_text = np.loadtxt(graph_file, dtype=np.int)
    num_rows = raw_text[0]
    num_cols = raw_text[1]

    offset = 2 + num_rows + 1
    I = raw_text[2 : offset]
    
    nnz = I[num_rows]

    J = raw_text[offset : offset + nnz]

    offset += nnz
    Data =  raw_text[offset : offset + nnz]

    vertex_edge = csr_matrix((Data, J, I), shape=(num_rows, num_cols))
    adjacency_mat = vertex_edge.dot(vertex_edge.transpose())
    
    return nx.from_scipy_sparse_matrix(adjacency_mat)

def convert_binary(node_value):
    node_color_value = ['r'] * len(node_value)
    for i in range(len(node_value)):
        node_color_value[i] = 'b' if node_value[i] > 0.0 else 'y'
    return node_color_value

def main(argv):
    graph = read_graph(argv[0])
    pos = np.loadtxt(argv[1])[1:]
    node_color_value = convert_binary(np.loadtxt(argv[2]))
    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(graph,pos,alpha=0.08)
    nx.draw_networkx_nodes(graph,pos,
                           node_size=0.02,
                           node_color=node_color_value)
                           # cmap=plt.cm.RdYlGn)

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    exit(main(sys.argv[1:]))
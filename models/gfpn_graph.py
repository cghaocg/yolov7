import torch
import torch.nn as nn
import numpy as np
import dgl
from dgl.nn import AvgPooling

from models.common import *

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')

def stochastic_create_edges(g, n_edges = 0):
    assert n_edges > g.num_nodes(), "number of edges is smaller than that of nodes"

    # It seems that CPU works faster than GPU in edges creations
    max_edges = (g.num_nodes()-1) * g.num_nodes() // 2  # (0 + g.num_nodes()-1) * g.num_nodes() / 2 

    # Ensure that every nodes has least one edge 
    for i in range(1, g.num_nodes()-1):
        j = np.random.randint(i+1, g.num_nodes())
        g.add_edges(torch.tensor([i], dtype=torch.int64), torch.tensor([j], dtype=torch.int64))
    
    # Add the reset of edges
    if n_edges:
        while g.num_edges() < n_edges and g.num_edges() < max_edges:
            i = np.random.randint(0, g.num_nodes())
            j = np.random.randint(0, g.num_nodes())
            g.add_edges(torch.tensor([i], dtype=torch.int64), torch.tensor([j], dtype=torch.int64)) if not (g.has_edges_between(i,j) or g.has_edges_between(j,i) or i == j) else 0
            if g.num_edges() == max_edges:
                break

    return dgl.add_reverse_edges(g, copy_ndata = True)


def heterograph(name_n_feature, dim_n_feature, nb_nodes = 2, is_birect = True):
    graph_data = {
        ('n', 'contextual', 'n'): (torch.tensor([0]), torch.tensor([1])),
        ('n', 'hierarchical', 'n'): (torch.tensor([0]), torch.tensor([1]))
    }
    g = dgl.heterograph(graph_data, num_nodes_dict = {'n': nb_nodes})
    g.nodes['n'].data[name_n_feature] = torch.zeros([g.num_nodes(), dim_n_feature], requires_grad=True)
    if is_birect:
        device = g.device
        print('device:', device)
        g = g.to(cpu)
        g = dgl.to_bidirected(g, copy_ndata = True)
        g = g.to(device)

    return g


def hetero_add_edges(g, u, v, edges):
    if isinstance(u,int):
        g.add_edges(torch.tensor([u]), torch.tensor([v]), etype = edges)
    elif isinstance(u,list):
        g.add_edges(torch.tensor(u), torch.tensor(v), etype = edges)
    else:
        g.add_edges(u, v, etype = edges)
    return g


def neighbor_9(i, c_shape):
    return torch.tensor([i-c_shape-1, i-c_shape, i-c_shape+1, i-1, i, i+1, i+c_shape-1, i+c_shape, i+c_shape+1])


def neighbor_25(i, c_shape):
    return torch.tensor([i-2*c_shape-2, i-2*c_shape-1, i-2*c_shape, i-2*c_shape+1, i-2*c_shape+2,
                        i-c_shape-2, i-c_shape-1, i-c_shape, i-c_shape+1, i-c_shape+2, 
                        i-2, i-1, i, i+1, i+2, 
                        i+c_shape-2, i+c_shape-1, i+c_shape, i+c_shape+1, i+c_shape+2,
                        i+2*c_shape-2, i+2*c_shape-1, i+2*c_shape, i+2*c_shape+1, i+2*c_shape+2])


def simple_graph(g):
    device = g.device
    g = g.to(cpu)
    g = dgl.to_simple(g, copy_ndata = True)
    g = g.to(device)
    return g


def to_birected(g):
    device = g.device
    g = g.to(cpu)
    g = dgl.to_bidirected(g, copy_ndata = True)
    g = g.to(device)
    return g


def simple_birected(g):
    device = g.device
    g = g.to(cpu)
    g = dgl.to_simple(g, copy_ndata = True)
    g = dgl.to_bidirected(g, copy_ndata = True)
    g = g.to(device)
    return g


# local pooling based on neighbor nodes, worked but it slows the training loop
def avg_pool_local(g, etype):
    for node in range(g.num_nodes()):
        _, neighbor = g.out_edges(node, form='uv', etype = etype)  # return srcnodes and dstnodes
        # local_g = g.out_subgraph({"n" : [node]})
        local_g = dgl.node_subgraph(g, neighbor)
        # print(local_g.ndata["pixel"])
        avg_pool = AvgPooling()
        h = avg_pool(local_g, local_g.ndata["pixel"])
        g.apply_nodes(lambda nodes: {'pixel' : h}, v = node)

        # h = dgl.nn.tensorflow.glob.AvgPooling(local_g, )
        # print(h)
        # pdb.set_trace()
        # _, neighbor = g.out_edges(node, form='uv', etype = etype)  # return srcnodes and dstnodes
        # neighbor_data = tf.gather(g.ndata["pixel"], neighbor)
        # mean = tf.expand_dims(tf.reduce_mean(neighbor_data, axis = 0), axis = 0)
        # g.apply_nodes(lambda nodes: {'pixel' : mean}, v = node)
    return g
        
    

def build_edges(g, c3_shape = 32, c4_shape = 16, c5_shape = 8):
    c3_size, c4_size , c5_size = c3_shape * c3_shape, c4_shape * c4_shape, c5_shape * c5_shape

    c3 = torch.arange(0, c3_size)
    c4 = torch.arange(c3_size, c3_size + c4_size)   
    c5 = torch.arange(c3_size + c4_size, c3_size + c4_size + c5_size)
    
    # build contextual edges
    for i in range(c3_shape - 1):
        g = hetero_add_edges(g, c3[i*c3_shape : (i+1)*c3_shape], c3[(i+1)*c3_shape : (i+2)*c3_shape], 'contextual')          # build edges between different rows (31 * 32 = 992)
        g = hetero_add_edges(g, c3[i : (c3_size+i) : c3_shape], c3[i+1 : (c3_size+i+1) : c3_shape], 'contextual')            # build edges between different column (31 * 32 = 992)
    for i in range(c4_shape - 1):
        g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c4[(i+1)*c4_shape : (i+2)*c4_shape], 'contextual')          # 15 * 16 = 240
        g = hetero_add_edges(g, c4[i : (c4_size+i) : c4_shape], c4[i+1 : (c4_size+i+1) : c4_shape], 'contextual') 
        # g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c3)
    for i in range(c5_shape - 1):
        g = hetero_add_edges(g, c5[i*c5_shape : (i+1)*c5_shape], c5[(i+1)*c5_shape : (i+2)*c5_shape], 'contextual')          # 7 * 8 = 56
        g = hetero_add_edges(g, c5[i : (c5_size+i) : c5_shape], c5[i+1 : (c5_size+i+1) : c5_shape], 'contextual') 
    
    # build hierarchical edges
    # peter, wait to modify
    c3_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape:2, 2:c3_shape:2]  # Get pixel indices in C3 for build hierarchical edges
    c4_stride = torch.reshape(c4, (c4_shape, c4_shape))[2:c4_shape:2, 2:c4_shape:2]
    c5_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape-4:4, 2:c3_shape-4:4]
    print('c3_stride.shape:', c3_stride.shape, 'c4_stride.shape:', c4_stride.shape, 'c5_stride.shape:', c5_stride.shape)

    
    # Edges between c3 and c4
    counter = 1
    for i in torch.reshape(c3_stride, [-1]).numpy():
        c3_9 = neighbor_9(i, c3_shape)
        g = hetero_add_edges(g, c3_9, c4[counter], 'hierarchical') 
        if counter % (c4_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1

    # Edges between c4 and c5
    counter = 1
    for i in torch.reshape(c4_stride, [-1]).numpy():
        c4_9 = neighbor_9(i, c4_shape)
        g = hetero_add_edges(g, c4_9, c5[counter], 'hierarchical') 
        if counter % (c5_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1
    
    # Edges between c3 and c5
    counter = 1
    for i in torch.reshape(c5_stride, [-1]).numpy():
        c5_9 = neighbor_25(i, c3_shape)
        g = hetero_add_edges(g, c5_9, c5[counter], 'hierarchical') 
        if counter % (c5_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1
    return g


def nodes_update(g, val):
    g.apply_nodes(lambda nodes: {'pixel' : val})


def hetero_subgraph(g, edges):
    return dgl.edge_type_subgraph(g, [edges])


def cnn_gnn(g, c):
    #print('g.ndata["pixel"]:', len(g.ndata["pixel"]), 'c:', len(c))
    #print('g.device:', g.device, 'c.device:', c.device)
    # for code test, skip first check
    if (len(g.ndata["pixel"]) != len(c)):
        return g
    else:
        g.ndata["pixel"] = c
    return g


def gnn_cnn(g):
    batch = len(g.ndata["pixel"]) // 1344   # 1344 = (32^2) + (16^2) + (8^2)
    num_p3, num_p4, num_p5 = 32 * 32 * batch, 16 * 16 * batch, 8 * 8 * batch
    p3 = torch.reshape(g.ndata["pixel"][:num_p3], (batch, 32, 32, 256))    # number of pixel in layers p3, 32*32 = 1024
    p4 = torch.reshape(g.ndata["pixel"][num_p3:num_p3+num_p4], (batch, 16, 16, 256)) # number of pixel in layers p4, 16*16 = 256
    p5 = torch.reshape(g.ndata["pixel"][num_p3+num_p4:num_p3+num_p4+num_p5], (batch, 8, 8, 256))  # number of pixel in layers p5, 8*8 = 64
    p3 = torch.permute(p3, [0, 3, 1, 2])
    p4 = torch.permute(p4, [0, 3, 1, 2])
    p5 = torch.permute(p5, [0, 3, 1, 2])
    return [p3, p4, p5]


if __name__ == "__main__":

    g = heterograph("pixel", 256, 1344, is_birect = False)
    g = simple_birected(build_edges(g))
    # equivalent to "tf.random.uniform()"
    minval, maxval = -10, 10
    g.ndata["pixel"] = (maxval - minval) * torch.rand([g.num_nodes(), 256]) + minval
    c_layer = ContextualLayers(256, 256)
    subc = hetero_subgraph(g, "contextual")
    subh = hetero_subgraph(g, "hierarchical")
    nodes_update(subc, c_layer(subc, subc.ndata["pixel"]))
    print(subc.ndata["pixel"])
    print(g.ndata["pixel"])
    print(subh.ndata["pixel"])
    # g = avg_pool_local(sub_c, "contextual")
    # starttime = datetime.datetime.now()
    # g1 = dgl.graph(([0], [1]), num_nodes = 4096)

    # g1 = stochastic_create_edges(g1,100000)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)

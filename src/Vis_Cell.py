import pydot
import os
from IPython.display import Image, display
import numpy as np

num_layers = 8

fns = ['tanh', 'sig', 'id', 'zero']
ops = ['add', 'multiply']

#arc_seq = [con_1, fn_1, con_2, fn_2, op_1 ,....,]
"""
arch_seq =[0, 1, 0, 1, 1,
           1, 0, 1, 0, 1,
           0, 1, 1, 1, 0,
           2, 0, 2, 0, 0,
           3, 1, 2, 1, 1,
           4, 0, 2, 0, 0,
           2, 1, 5, 1, 1,
           6, 0, 4, 0, 1,
           7, 1, 6, 1, 1,
           ]
"""
#arch_seq = np.array(arch_seq[20:])
arch_seq = np.array([1, 0, 2, 1, 1,
                     2, 1, 3, 2, 1,
                     4, 2, 5, 2, 0,
                     0, 1, 6, 0, 1,
                     ])

callgraph = pydot.Dot(graph_name="Cell", graph_type="digraph")
cluster = pydot.Cluster(name="pre_process", label='Pre_processing')

x = pydot.Node("x", style = 'filled', fillcolor = 'yellow')
p_c = pydot.Node("p_c",style = 'filled', fillcolor = 'pink')
p_h = pydot.Node("p_h",style = 'filled', fillcolor = 'skyblue')

cluster.add_node(x)
cluster.add_node(p_h)
cluster.add_node(p_c)

add = pydot.Node(ops[0])
cluster.add_node(add)
multiply = pydot.Node(ops[1])
#cluster.add_node(multiply)

edge_a = pydot.Edge(p_h, add, label= "w_h")
cluster.add_edge(edge_a)
edge_b = pydot.Edge(x, add, label= "w_x")
cluster.add_edge(edge_b)

"""
edge_c = pydot.Edge(p_c, add)
cluster.add_edge(edge_c)
edge_d = pydot.Edge(multiply, add)
cluster.add_edge(edge_d)
"""
callgraph.add_subgraph(cluster)

Layers = {}
Nodes = {}
Edges = {}
hid = {}
ops_dict = {}
int_edges = {}
start_idx = 0
for i in range(num_layers):
    print("i = {} ".format(i))
    if i < 4:
        Layers['cluster_{}'.format(i)] = pydot.Cluster(name= "Layer_{}".format(i), label= "Layer_{}".format(i))
        Nodes['Node_{}'.format(i)] = pydot.Node(name= "Node_{}".format(i), style = "filled", fillcolor = "purple")
        Layers['cluster_{}'.format(i)].add_node(Nodes['Node_{}'.format(i)])
        Layers['cluster_{}'.format(i)].add_node(add)
        Edges['edge_{}'.format(i)] = pydot.Edge("add", str(Nodes['Node_{}'.format(i)]))
        Layers['cluster_{}'.format(i)].add_edge(Edges['edge_{}'.format(i)])
        #callgraph.add_subgraph(Layers['cluster_{}'.format(i)])
        #callgraph.add_edge(Edges['edge_{}'.format(i)])

    else:
        Layers['cluster_{}'.format(i)] = pydot.Cluster(name = "Layer_{}".format(i), label = 'Layer_{}'.format(i))
        hid["h_{}_1".format(i)] = pydot.Node(name= "h_{}_1".format(i))
        hid["h_{}_2".format(i)] = pydot.Node(name= "h_{}_2".format(i))
        Layers['cluster_{}'.format(i)].add_node(hid["h_{}_1".format(i)])
        Layers['cluster_{}'.format(i)].add_node(hid["h_{}_2".format(i)])

        if arch_seq[i + start_idx - 4] > 3:
            Edges['edge_{}_1'.format(i)] = pydot.Edge(ops_dict['Op_{}'.format(arch_seq[i + start_idx - 4])], hid['h_{}_1'.format(i)])
        else:
            Edges['edge_{}_1'.format(i)] = pydot.Edge(Nodes['Node_{}'.format(arch_seq[i + start_idx - 4])], hid['h_{}_1'.format(i)])

        if arch_seq[i + start_idx - 2] > 3:
            Edges['edge_{}_2'.format(i)] = pydot.Edge(ops_dict['Op_{}'.format(arch_seq[i + start_idx - 2])], hid['h_{}_2'.format(i)])
        else:
            Edges['edge_{}_2'.format(i)] = pydot.Edge(Nodes['Node_{}'.format(arch_seq[i + start_idx - 2])], hid['h_{}_2'.format(i)])

        callgraph.add_edge(Edges['edge_{}_1'.format(i)])
        callgraph.add_edge(Edges['edge_{}_2'.format(i)])

        ops_dict['Op_{}'.format(i)] = pydot.Node(name = str(ops[arch_seq[i + start_idx]]) + "_{}".format(i))
        Layers['cluster_{}'.format(i)].add_node(ops_dict["Op_{}".format(i)])
        print("Op_{} : ".format(i), ops_dict["Op_{}".format(i)] )
        int_edges['Edge_{}_1'.format(i)] = pydot.Edge(hid['h_{}_1'.format(i)], ops_dict["Op_{}".format(i)],
                                                      label=fns[arch_seq[i + start_idx - 3]])
        int_edges['Edge_{}_2'.format(i)] = pydot.Edge(hid['h_{}_2'.format(i)], ops_dict["Op_{}".format(i)],
                                                      label=fns[arch_seq[i + start_idx  - 1]])

        #Layers['cluster_{}'.format(i)].add_edge(int_edges['Edge_{}_1'.format(i)])
        #Layers['cluster_{}'.format(i)].add_edge(int_edges['Edge_{}_2'.format(i)])
        Layers['cluster_{}'.format(i)].add_edge(int_edges['Edge_{}_2'.format(i)])
        Layers['cluster_{}'.format(i)].add_edge(int_edges['Edge_{}_1'.format(i)])
        callgraph.add_subgraph(Layers['cluster_{}'.format(i)])
        start_idx += 4


callgraph.add_node(pydot.Node("ct", style = "filled", fillcolor= "pink"))
callgraph.add_edge(pydot.Edge(ops_dict["Op_6"], 'ct'))

callgraph.add_node(pydot.Node("ht", style = "filled", fillcolor= "skyblue"))
callgraph.add_edge(pydot.Edge(ops_dict['Op_7'], 'ht'))

callgraph.write_png("model.png")
"""
im = Image(callgraph.create_png())
display(im)
open('image.png', 'wb').write(im.data)

"""


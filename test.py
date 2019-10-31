import json
import random
import os
# import sys
# sys.path.append('.')
from graph.graph import Graph
from graph.ternary import Tree

def main():
    with open('data/small_dataset.json', 'r') as f:
        tiny_dataset = json.load(f)
    sampled_id = random.randint(0, len(tiny_dataset) - 1)
    data = tiny_dataset[sampled_id]
    dependencies = data['dependency']

    g = Graph()
    for dep in dependencies:
        gov_node = g.add_node(dep['governorGloss'], dep['governor'], "")
        dep_node = g.add_node(dep['dependentGloss'], dep['dependent'], "")
        g.add_edge(gov_node, dep_node, dep['dep'])

    tree = g.to_tree()
    print(str(g))
    with open(os.path.join('tmp', 'graph.dot'), 'w') as f:
        f.write(g.graphviz())
    with open(os.path.join('tmp', 'tree.dot'), 'w') as f:
        f.write(tree.graphviz())

if __name__ == '__main__':
    main()

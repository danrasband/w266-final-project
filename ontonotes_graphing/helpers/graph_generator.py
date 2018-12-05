# Model
import spacy
import networkx as nx

# Utils
from tqdm import tqdm

def generate_graph(entities, sentences, number_of_generations=2):
    '''Takes a DataFrame of entities and converts them into a weighted, 
    directed graph of types, relations, children, etc.'''
    graphs_dict = dict()
    for entity_id, entity in entities.iterrows():
        graph_row(graphs_dict, entity, sentences, number_of_generations)
    return graphs_dict

def graph_row(graphs_dict, entity, sentences, number_of_generations=2):
    '''Takes a Series (a single row of a DataFrame) and adds its entity information 
    to the graphs.'''
    root_node = str(entity['type'])

    # Retrieve our Directed Graph for this NE Type or create a new one
    G = graphs_dict.get(root_node, nx.DiGraph())
    
    # For each row, add a node for the Named Entity's type
    G.add_node(root_node)
    node_weight = G.nodes[root_node].get('weight', 0)
    G.node[root_node]['weight'] = node_weight + 1
    G.node[root_node]['xlabel'] = G.node[root_node]['weight']

    # Let's assume the head is the last word of the phrase
    # and get that Token from the spaCy parse:
    
    spacy_parsed = sentences.loc[entity.sentence_id]['spacy_parsed']
    head_index = int(entity['end_index'])
    head_of_phrase = spacy_parsed[head_index]
    
    nodes_needing_head_branches = [(head_of_phrase, root_node)]
    next_head_nodes = []
    nodes_needing_child_branches = [(head_of_phrase, root_node)]
    next_child_nodes = []
    
    current_gen = 1
    
    while current_gen <= number_of_generations:
        
        for (node, child_label) in nodes_needing_head_branches:
            try: 
                # Get the explanation of its dependency type in this usage
                relation = spacy.explain(node.dep_)

                # If no explanation, revert to the raw dependency type.
                if relation is None:
                    relation = node.dep_

                # Trying to catch and diagnose some problem cases
                elif relation == 'punctuation':
                    print('NE \'{1}\' marked as punctuation in sentence \'{0}\''.format(str(spacy_parsed), entity['string']))
                    print(' --- ')
                elif relation == 'determiner':
                    print('NE \'{1}\' marked as determiner in sentence \'{0}\''.format(str(spacy_parsed), entity['string']))
                    print(' --- ')

                # Object of preposition doesn't do much, so let's see what's on the other side of that.
                elif relation == 'object of preposition':
                    relation = 'head of prep phrase'
                    # move to the preposition so we get its head later on when adding node
                    node = node.head
                
                # differentiating head-focused edges from child-focused edges
                intermediary_node_label = 'head g{0} {1}'.format(current_gen, relation)

                # Add a node for the relation, and connect that to the main entity.
                # Add the weights to the nodes and edges.
                G.add_node(intermediary_node_label)
                relation_weight = G.node[intermediary_node_label].get('weight', 0)
                G.node[intermediary_node_label]['weight'] = relation_weight + 1
                
                # Add a node from the dependency type to the entity's head and add weights
                norm = node.head.norm_
                G.add_node(norm)
                norm_edge_weight = G.node[norm].get('weight', 0)
                G.node[norm]['weight'] = norm_edge_weight + 1
                
                G.add_edge(child_label, intermediary_node_label, label='head')
                
                G.add_edge(intermediary_node_label, norm)
                
                # Add the next round for the next generation
                if node.head != node:
                    next_head_nodes.append((node.head, norm))
                
            except:
                print('passed in head')
                pass
            
            # Move the next round into the queue and clear it
            nodes_needing_head_branches = next_head_nodes
            next_head_nodes = []

            for (node, parent_label) in nodes_needing_child_branches:
                for child in node.children:

                    # Get the relation of the child node to this one.
                    relation = spacy.explain(child.dep_)

                    # If no explanation, revert to the raw dependency type.
                    if relation is None:
                        relation = child.dep_

                    if relation == 'punctuation':
                        break

                    # Differentiate these relations from head relations
                    # and add the node and its weights
                    intermediary_node_label = 'child g{0} {1}'.format(current_gen, relation)
                    G.add_node(intermediary_node_label)
                    relation_weight = G.nodes[intermediary_node_label].get('weight', 0)
                    G.node[intermediary_node_label]['weight'] = relation_weight + 1
                    
                    # Add the child as normed, and add its edge and weights
                    child_norm = child.norm_
                    G.add_node(child_norm)
                    leaf_weight = G.node[child_norm].get('weight', 0)
                    G.node[child_norm]['weight'] = leaf_weight + 1

                    # add edge between the parent node and this relation and weights
                    G.add_edge(parent_label, intermediary_node_label, label='child')
                    
                    G.add_edge(intermediary_node_label, child_norm)

                    # Queue up the children for the next generation
                    for childs_child in child.children:
                        next_child_nodes.append((childs_child, child_norm))      
                
                # Move the children into the queue and clear it.
                nodes_needing_child_branches = next_child_nodes
                next_child_nodes = []
        
        # Increment the generation
        current_gen += 1

    graphs_dict[root_node] = G
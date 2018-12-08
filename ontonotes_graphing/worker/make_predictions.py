import boto3
import botocore
import spacy
import en_coref_lg
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import sys
import re
import dill as pickle
from collections import defaultdict

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_BUCKET = os.getenv('AWS_BUCKET')

NETS_PICKLE_FILEPATH = './data/LP_NET_Graphs.20181205.pkl'
OUTPUT_FOLDER = './data'
DOCUMENTS_FILEPATH = './data/documents_dev.pkl'

nlp = en_coref_lg.load()

# ### Graphing a Candidate Named Entity
# Defining helper functions to build the candidate graphs

# Defining helper functions to build the candidate graphs

def reconcile_ents_and_clusters(document_id, doc):
    """"Reconcile the coreference and entities lists into a
        a single dict of graphs to make.

        Keys are ent_id strings [document_id]:[start_word_index]:[end_word_index].
        Values are (spaCy.Span, graph_id) tuples."""

    # Keys are (start.idx, end.idx) tuples.
    # Values are (spaCy.Span, graph_id) tuples."
    occurence_ind  = {}

    cluster_offset = 0
    if doc._.has_coref:
        cluster_offset = len(doc._.coref_clusters)
        for cluster_idx, cluster in enumerate(doc._.coref_clusters):
            for mention in cluster:
                key = ":".join([document_id,str(mention.start), str(mention.end -1)])
                occurence_ind[key] = (mention, cluster_idx)

    # Now let's see if each ent is in there. If not, we'll add it to
    # our cluster list.
    new_cluster_idx = 0

    for ent_ind, ent in enumerate(doc.ents):
        key = ":".join([document_id,str(ent.start), str(ent.end -1)])
        try:
            occurence_ind[key]
        except:
            occurence_ind[key] = (ent, cluster_offset + new_cluster_idx)
            new_cluster_idx += 1

    print("Length of occurence index for doc {0} is: {1}".format(document_id, len(occurence_ind)))

    return occurence_ind

def graph_entity(ent, doc, G, root_node):

    NO_OF_GENERATIONS = 2

    # Assume the head of the phrase, if it is a phrase, is the last word
    # in the phrase.
    head_of_phrase = ent[-1]

    nodes_needing_head_branches = [(head_of_phrase, root_node)]
    next_head_nodes = []
    nodes_needing_child_branches = [(head_of_phrase, root_node)]
    next_child_nodes = []

    current_gen = 1
    while current_gen <= NO_OF_GENERATIONS:

        for (node, child_label) in nodes_needing_head_branches:
            try:
                # Get the explanation of its relation arc in this usage
                relation = spacy.explain(node.dep_)
                # If no explanation, revert to the raw dependency type.
                if relation is None:
                    relation = node.dep_

                if relation == 'ROOT':
                    continue

                # Object of preposition doesn't do much, so let's see what's on the other side of that.
                elif relation == "object of preposition":
                    relation = "head of prep phrase"
                    # move to the preposition so we get its head later on when adding node
                    node = node.head

                intermediary_node_label = "head g{0} {1}".format(current_gen, relation)

                # Add a node for the relation, and connect that to the main entity
                G.add_node(intermediary_node_label)
                G.add_edge(child_label, intermediary_node_label, label="head")

                # Add a node from the relation to the entity's head, and connect that
                # to the relation type
                normed_head = node.head.norm_
                #print("adding head node: {0}".format(normed_head))
                G.add_node(normed_head)
                G.add_edge(intermediary_node_label, normed_head)

                if node.head != node:
                    next_head_nodes.append((node.head, node.head.text))
            except Exception as ex:
                print("passed in head.\n{0}".format(ex))
                pass

        nodes_needing_head_branches = next_head_nodes
        next_head_nodes = []

        for (node, parent_label) in nodes_needing_child_branches:
            try:
                for child in node.children:
                    relation = spacy.explain(child.dep_)
                    # If no explanation, revert to the raw dependency type.
                    if relation is None:
                        relation = node.dep_

                    elif relation == 'punctuation':
                        continue

                    intermediary_node_label = "child g{0} {1}".format(current_gen, relation)

                    G.add_node(intermediary_node_label)
                    G.add_edge(parent_label, intermediary_node_label, label="child")

                    normed_child = child.norm_
                    G.add_node(normed_child)
                    #print("adding child node: {0}".format(normed_child))
                    G.add_edge(intermediary_node_label, normed_child)

                    for childs_child in child.children:
                        next_child_nodes.append((childs_child, childs_child.text))
            except Exception as ex:
                print("passed in child.\n{0}".format(ex))
                pass
        nodes_needing_child_branches = next_child_nodes
        next_child_nodes = []

        # Increment the generation
        current_gen += 1

    return G

def graph_candidates_in_doc(document_id, candidate_text):

    doc = nlp(candidate_text)

    # Key of clustered_ents is ent_it. Value is (Span, graph_cluter_id)
    clustered_ents = reconcile_ents_and_clusters(document_id, doc)

    # Initialize a graph for each clustered_ent
    candidate_graphs = dict()

    for ent_id, (ent, graph_idx) in clustered_ents.items():

        # Make sure we have our root. No harm done if it already exists.
        # If it's a cluster, we get the Span of the most representative
        # mention in the cluster
        try:
            root_node = doc._.coref_clusters[graph_idx].main.text
        # If it's not, we just use the ent name
        except:
            root_node = ent.text

        # Get the cluster's root node and existing graph from previous mentions
        # or create a new tuple of those.
        root_node, G, _ = candidate_graphs.get(graph_idx, (root_node, nx.DiGraph(), graph_idx))

        G.add_node(root_node)

        # A helper function adds the rest of the graph
#         print("\nGraphing entity: {0}".format(ent.text))
        candidate_graphs[graph_idx] = (root_node, graph_entity(ent, doc, G, root_node), graph_idx)

    spacy_ent_graphs = {}

    # Filter back to just the spaCy ents
    for ent in doc.ents:
        key = ":".join([document_id,str(ent.start), str(ent.end - 1)])
        _, ent_graph_idx = clustered_ents[key]
        spacy_ent_graphs[key] = candidate_graphs[ent_graph_idx]

    #graph_entity(ent, doc) for ent in doc.ents]
    return spacy_ent_graphs

def logsumexp(a):
    """Simple re-implementation of scipy.misc.logsumexp."""
    a_max = np.max(a)
    if a_max == -np.inf:
        return -np.inf
    sumexp = np.sum(np.exp(a - a_max))
    return np.log(sumexp) + a_max

def score_relation_and_children(sim_nlp, cand_G, NET_G, cand_parent, net_options, cand_succ_dict, NET_succ_dict, sim_dict): #, parent_weight):

    # If this node has no further children,
    # compare it to its NET options
    if cand_parent not in cand_succ_dict:

        sim_scores = list()
        cand_token = sim_nlp(cand_parent)[0]

        for net_opt in net_options:
            try:
                sim_scores.append(sim_dict[cand_parent][net_opt])
            except:
                score = cand_token.similarity(sim_nlp(net_opt)[0])
                sim_scores.append(score)
                sim_dict[cand_parent][net_opt] = score

        # Get the index of the most similar word
        sim_idx = np.argmax(sim_scores)

        # Recover the float value of the winning word's weight
        #sim_weight = float(NET_G.node[net_options[sim_idx]]['weight'])

        # Score the similarity times the square root of the word's frequency (weight)
        similarity_score = sim_scores[sim_idx] #* (sim_weight / parent_weight) #np.log(sim_scores[sim_idx]) #+ sim_lp

        return similarity_score

    # Otherwise let's score the dependency tags and
    # recursively call this on their children
    else:
        # Prepare to hold scores from multiple branches
        accumulated_scores = []

        # Iterate over dependency relations from the parent
        for relation in cand_succ_dict[cand_parent]:

            # Proceed if the NET_graph has this relation:
            try:
                # Get the options from the NET graph branching from this relation type
                child_net_options = NET_succ_dict[relation]
                #relation_weight = float(NET_G.node[relation]['weight'])
                #print('relation weight:' + str(relation_weight))

                # Iterate over the children of each relation
                for cand_child in cand_succ_dict[relation]:
                    score_from_this_node_to_leaves = score_relation_and_children(nlp, cand_G, NET_G, cand_child, child_net_options, cand_succ_dict, NET_succ_dict, sim_dict) #, relation_weight)
                    if score_from_this_node_to_leaves is not None:
                        accumulated_scores.append(score_from_this_node_to_leaves) # * (np.log(relation_weight) / np.log(parent_weight)))

            except Exception as ex:
                print(ex)

        # If we have more than an empty list
        if accumulated_scores != list():
            return sum(accumulated_scores) # logsumexp(accumulated_scores)


def compare_candidate_to_NET(nlp, candidate_G, candidate_root, NET_G, net_root, sim_dict):

    # Calculate the breadth-first search of the candidate graph
    cand_succ_dict = {par:child_list for (par,child_list)in nx.bfs_successors(candidate_G, candidate_root)}

    # Calculate the breadth-first search of the NET graph
    # (For a speed boost we should do this externallay and pass it in.)
    NET_succ_dict = {par:child_list for (par,child_list)in nx.bfs_successors(NET_G, net_root)}

    print("\n\nNET: {0}".format(net_root))

    # Run the results of runnign the recursive score_relation_and_children function on the initial roots
    return score_relation_and_children(nlp, candidate_G, NET_G, candidate_root, [], cand_succ_dict, NET_succ_dict, sim_dict) #, NET_G.node[net_root]['weight'])


# ### Lop through all the NETs and predict the Max

def predict_max_from_all_nets(cand_root_node, cand_G, NETs_dict, nlp, sim_dict):

    # Initialize dict to hold the scores
    similarity_scores_dict = dict()

    # Iterate through the NETs
    for net_string, net_graph in NETs_dict.items():

        # Score the candidate against each NET
        sim_score = compare_candidate_to_NET(nlp, cand_G, cand_root_node, net_graph, net_string, sim_dict)

        # If there is a score for this NET, store it
        if sim_score is not None:
            similarity_scores_dict[net_string] = sim_score
    print(" *** ")
    print(cand_root_node)
    print(similarity_scores_dict)

    types_in_likelihood_order = ['ORG', 'GPE', 'PERSON', 'DATE', 'CARDINAL',
                                 'NORP', 'MONEY', 'PERCENT', 'ORDINAL', 'LOC',
                                 'TIME', 'WORK_OF_ART', 'QUANTITY', 'FAC',
                                 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE']

    high_score = float("-inf")
    prediction_so_far = 'NO_MATCH'

    for key,value in similarity_scores_dict.items():
        if value > high_score:
            high_score = value
            prediction_so_far = key

    return prediction_so_far

def predict_on_doc(doc_row, NETs_dict, nlp, Y_pred, sim_dict):

    cluster_dict = dict()

    removed_pw = re.sub(r'%pw', 'unk_date', doc_row["document"])
    removed_hypenation = re.sub(r'([a-z])-([a-z])', '\1\2', removed_pw)

    filtered_doc = removed_hypenation

    start_time = time.time()
    print('Document ID: {}'.format(doc_row.name))
    ent_dict = graph_candidates_in_doc(doc_row.name,filtered_doc)
    print('Candidates in doc took {} seconds.'.format(time.time() - start_time))

    for ent_id, (cand_root_node, cand_G, cluster_id) in ent_dict.items():
        try:
            Y_pred.append([ent_id, cluster_dict[cluster_id]])
        except:
            cluster_dict[cluster_id] = predict_max_from_all_nets(cand_root_node, cand_G, NETs_dict, nlp, sim_dict)
            Y_pred.append([ent_id, cluster_dict[cluster_id]])


# ### Loop through documents to generate Y_Pred entitites

def make_predictions(first_doc, last_doc):

    ### Import the documents with trace strings removed
    with open(DOCUMENTS_FILEPATH, 'rb') as file:
        documents = pickle.load(file)

    with open(NETS_PICKLE_FILEPATH, 'rb') as file:
        lp_net_graphs = pickle.load(file)

    checkpoint_1 = time.time()

    DOC_MIN = first_doc
    DOC_MAX = last_doc

    # Build the graphs of candidates from the documents.
    sim_dict = defaultdict(lambda: dict())

    # Loading a spaCy model but disabling parsers to speed up similarity measurements
    # sim_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner', 'neuralcoref'])
    sim_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
    #sim_nlp.add_pipe(sim_nlp.create_pipe('sentencizer'))

    for i in range(DOC_MIN, DOC_MAX + 1):
        filename = 'Y_pred_{}.csv'.format(i)
        if exists_on_s3(filename): continue

        document = documents.iloc[i]
        print(document)

        Y_pred = list()
        predict_on_doc(document, lp_net_graphs, sim_nlp, Y_pred, sim_dict)

        ypred_df = pd.DataFrame(Y_pred, columns = ['entity_id', 'prediction']).set_index('entity_id')
        csv_output_file_path = '{}/{}'.format(OUTPUT_FOLDER, filename)
        ypred_df.to_csv(csv_output_file_path)

        save_to_s3(csv_output_file_path, filename)

    print("That took: {} seconds".format(round(time.time() - checkpoint_1, 4)))


def save_to_s3(filepath, key):
    s3 = boto3.client('s3')
    with open(filepath, 'rb') as file:
        s3.put_object(
            Key=key,
            Bucket=AWS_BUCKET,
            Body=file,
        )

def exists_on_s3(key):
    try:
        s3 = boto3.resource('s3')
        s3.Object(AWS_BUCKET, key).load()
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise e


if __name__ == "__main__":

    make_predictions(int(sys.argv[1]), int(sys.argv[2]))

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# standard library imports
import argparse
import os
import pickle

# third party imports
from dotenv import load_dotenv
from py2neo import Graph, Node, NodeMatcher, Relationship
from py2neo import Database


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# module functions
def build_cluster_node(cluster_id: int,
                       kwargs,
                       ) -> Node:
    """Generate cluster node from MLN network using selected cluster ID"""
    cluster_name = 'cluster_{}'.format(cluster_id)
    cluster_node = Node('Cluster', name=cluster_name, **kwargs)
    return cluster_node


def make_cluster_properties(cluster_id: int,
                            cluster_dict: dict
                            ) -> dict:
    """Create the cluster properties to go with the cluster node"""
    return {'cluster_id': cluster_id,
            'cluster_total_count': cluster_dict['_ttlCnt'],
            'cluster_type': cluster_dict['_type'],
            'cluster_is_stop': cluster_dict['_isStop'],
            'cluster_next_cluster_id': cluster_dict['_nxtArgClustIdx'],
            }


# TODO
# Populate graph database with parts, tree nodes, etc.
# Refactor functions and graph database data ingest into a class
# Test using larger dataset


if __name__ == '__main__':
    """Populate the graph database with MLN data"""
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # define and load inputs
    parser = argparse.ArgumentParser(description='API wrapper on Dooblo\'s '
                                                 'SurveyToGo software.')
    parser.add_argument('-e', '--env_path', required=True, help='Path to .env file')
    parser.add_argument('-m', '--mln_dict_src', required=True, help='Path to pickled MLN dict.')
    parser.add_argument('-v', '--verbose', const=False, type=int, choices=[True, False],
                        help='Select verbosity: True to print cluster hierarchy')
    parser.add_argument('-x', '--delete_database', const=False, type=bool, choices=[True, False],
                        help='If true, delete neo4j db')
    args_dict = vars(parser.parse_args())
    # parse args_dict
    delete_database = args_dict['delete_database']
    env_path = args_dict['env_path']
    mln_dict_src = args_dict['mln_dict_src']
    verbose = args_dict['verbose']
    # import MLN, which requires that CORENLP_HOME dir is an environment variable
    load_dotenv(env_path)
    CORENLP_HOME = os.getenv('CORENLP_HOME')
    from multivac.pymln.semantic import MLN
    # define neo4j variables
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    db = Database(uri=NEO4J_URI, password=NEO4J_PASSWORD)
    graph = Graph(database=db)
    if delete_database:
        graph.delete_all()
    # instantiate graph
    node_matcher = NodeMatcher(graph)
    # load mln data
    MLN.load_mln(mln_dict_src)
    with open(mln_dict_src, 'rb') as f:
        mln_dict = pickle.load(f)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # populate the graph database with mln data
    for cluster_id, cluster_object in mln_dict['clusts'].items():
        if verbose:
            print('cluster_id: {}'.format(cluster_id))
        cluster_dict = cluster_object.__dict__
        cluster_properties = make_cluster_properties(cluster_id, cluster_dict)
        cluster_node = build_cluster_node(cluster_id, cluster_properties)
        # the tx - transaction - object writes to the graph database using the commit method
        tx = graph.begin()
        if cluster_id == 1:
            tx.create(cluster_node)
        else:
            tx.merge(cluster_node, primary_label='Cluster', primary_key='name')
        tx.commit()
        # get relationships between clusters via arg_clusters
        # arg_clusters are their own entities
        # each arg_cluster contains one or more tokens (e.g., dog)
        # each arg_cluster contains zero or more child clusters
        # each arg_cluster rolls up to one or more clusters

        arg_clusters_dict = {k: v.__dict__ for k, v in cluster_dict['_argClusts'].items()}
        for arg_cluster_id, arg_cluster_dict in arg_clusters_dict.items():
            if verbose:
                print('\targ_cluster_id: {}'.format(arg_cluster_id))
            arg_cluster_name = 'arg_cluster_{}'.format(arg_cluster_id)
            arg_cluster_kwargs = {'total_arg_cluster_count': arg_cluster_dict['_ttlArgCnt']}
            arg_cluster_node = Node('ArgCluster', name=arg_cluster_name, **arg_cluster_kwargs)
            # relate each  to cluster
            arg_cluster_relationship = Relationship(arg_cluster_node, 'ARG CLUSTER OF',
                                                    cluster_node)
            tx = graph.begin()
            tx.create(arg_cluster_relationship)
            # get relationships between clusters via arg_clusters
            child_clusters_dict = arg_cluster_dict['_chdClustIdx_cnt']
            tx.commit()
            for child_cluster_id, child_cluster_count in child_clusters_dict.items():
                tx = graph.begin()
                child_cluster_name = 'cluster_{}'.format(child_cluster_id)
                child_cluster_node = node_matcher.match('Cluster', name=child_cluster_name).first()
                if child_cluster_node is None:  # create node if it doesn't exist
                    child_cluster_dict = mln_dict['clusts'][child_cluster_id].__dict__
                    child_cluster_properties = make_cluster_properties(child_cluster_id,
                                                                       child_cluster_dict)
                    child_cluster_node = build_cluster_node(child_cluster_id,
                                                            child_cluster_properties)
                    tx.create(child_cluster_node)
                else:
                    tx.merge(child_cluster_node, primary_label='Cluster')
                cluster_cluster_kwargs = {'child_cluster_count': child_cluster_count}
                cluster_cluster_relationship = Relationship(arg_cluster_node,
                                                            'PARENT ARGUMENT OF',
                                                            child_cluster_node,
                                                            **cluster_cluster_kwargs)
                if verbose:
                    print('\t\tchild_cluster_id: {}'.format(child_cluster_id))
                tx.create(cluster_cluster_relationship)
                tx.commit()

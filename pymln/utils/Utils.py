

#
# Utility functions for pymln parsing
# 


def genTreeNodeID(aid, sid, wid):
    node_id = ':'.join([str(x) for x in [aid, sid, wid]])

    return node_id


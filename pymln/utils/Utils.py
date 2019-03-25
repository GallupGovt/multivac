

#
# Utility functions for pymln parsing
# 

import math

def inc_key(d, key, inc=1):
    if key not in d:
        d[key] = inc
    else:
        d[key] += inc

    return d

def dec_key(d, key, base=None, dec=1, remove=False):
    if key not in d:
        if base is None:
            d = None
        else:
            d[key] = base - dec
    else:
        d[key] -= dec

    if remove and d[key] <= 0:
        del d[key]

    return d


def genTreeNodeID(aid, sid, wid):
    node_id = '{0}:{1}:{2:03d}'.format(aid, sid, wid)

    return node_id


class java_iter(object):
    def __init__(self, it):
        self.it = iter(it)
        self._hasnext = None

    def __iter__(self): return self

    def next(self):
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.it)
            self._hasnext = None

        return result

    def hasnext(self):
        if self._hasnext is None:
            try: 
                self._thenext = next(self.it)
            except StopIteration: 
                self._hasnext = False
        else: 
            self._hasnext = True
    
        return self._hasnext


def compareStr(s, t):
    # compare each character until there's a difference!!!
        this = sum([ord(x) for x in s])
        that = sum([ord(x) for x in t])
        result = this - that

        return result


def xlogx(x):
    if x <= 0:
        result = 0
    else:
        result = x * math.log(x)

    return result


    


class Argument(object):
	def __init__(self, argNode, path, argPart):
		self._argNode = argNode
		self._path = path
		self._argPart = argPart

		return None

	def getPath(self):
		return self._path

	def getPart(self):
		return self._argPart

	def getNode(self):
		return self._argNode



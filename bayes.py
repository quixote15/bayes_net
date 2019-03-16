from Node import Node
import random

PLAYING_EXAMPLES = [(True, False), (True, False), (True, True),
                    (True, False), (True, False), (False, True),
                    (False, False), (False, True), (False, True),
                    (False, True), (False, False), (False, True),
                    (False, True)]

class BayesNet:
    # The nodes in the network
    nodes = []

    # Build the initial network
    def __init__(self):
        self.nodes.append(Node("Cloudy", [], [0.4]))
        self.nodes.append(Node("Drought", [], [0.002]))
        self.nodes.append(Node("Sprinkler",
                               [self.nodes[0], self.nodes[1]],
                               [0.02, 0.1, 0.15, 0.5]))
        self.nodes.append(Node("Rain", [self.nodes[0]], [0.8, 0.1]))
        self.nodes.append(Node("PlayOutside", [self.nodes[3]],
                          self.calculatePlayOutsideProbabilities(PLAYING_EXAMPLES)))
        self.nodes.append(Node("WetGrass",
                               [self.nodes[2], self.nodes[3]],
                               [0.99, 0.9, 0.9, 0.0]))

    # Prints the current state of the network to stdout
    def printState(self):
        strings = []
        for node in self.nodes:
            strings.append(node.name + " = " + str(node.value))

        print(", ".join(strings))

    def calculatePlayOutsideProbabilities(self, rainingInstances):
        playing = [0, 0]
        total = [0, 0]
        prob = [0.0, 0.0]

        for sample in rainingInstances:
            if sample[0]:
                playing[0] += 1 if sample[1] else 0
                total[0] += 1
            else:
                playing[1] += 1 if sample[1] else 0
                total[1] += 1

        prob[0] = float(playing[0]) / float(total[0])
        prob[1] = float(playing[1]) / float(total[1])

        return prob

    '''
    This method will sample the value for a node given its
    conditional probability.
    '''
    def sampleNode(self, node):
        node.value = True if random.random() <= node.conditionalProbability() else False

    '''
    This method assigns new values to the nodes in the network by
    sampling from the joint distribution.  Based on the PRIOR-SAMPLE
    from the text book/slides
    '''
    def priorSample(self):
        for n in self.nodes:
            self.sampleNode(n)

    '''
    This method will return true if all the evidence variables in the
    network have the value specified by the evidence values.
    '''
    def testModel(self, indicesOfEvidenceNodes, evidenceValues):
        for i in range(len(indicesOfEvidenceNodes)):
            if (self.nodes[indicesOfEvidenceNodes[i]].value != evidenceValues[i]):
                return False

        return True

    def printState(self):
        strings = []
        for node in self.nodes:
            strings.append(node.name + " = " + str(node.value))

        print(", ".join(strings))

if __name__ == "__main__":
    b = BayesNet()

    # Sample five state from joint distribution and print them
    for i in range(1000):
        b.priorSample()
        b.printState()

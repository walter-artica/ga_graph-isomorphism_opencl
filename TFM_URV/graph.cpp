#include "graph.h"
#include <random>
#include <algorithm>
#include <cmath>

inline int getAdjIndex(int nofVertices, int v1, int v2)
{
	if (v1 > v2) {
		int tmp = v1;
		v1 = v2;
		v2 = tmp;
	}
	return v1*nofVertices + v2 - (v1 + 1)*(v1 + 2) / 2;
}

Graph::Graph(int nofVertices) : nofVertices(nofVertices), nofEdges(0)
{
	degrees = new int[nofVertices]();
	adjMatrix = new bool[nofVertices*(nofVertices - 1) / 2]();
}

Graph::~Graph()
{
	delete[] degrees;
	delete[] adjMatrix;
}

bool Graph::addEdge(int v1, int v2)
{
	if (v1 == v2 || v1 < 0 || v2 < 0 || v1 >= nofVertices || v2 >= nofVertices)
		return false; // avoid loops and indices out of range
	int inx = getAdjIndex(nofVertices, v1, v2);
	if (adjMatrix[inx])
		return false;
	adjMatrix[inx] = true;
	degrees[v1]++;
	degrees[v2]++;
	nofEdges++;
	return true;
}

bool Graph::isEdge(int v1, int v2) const
{
	if (v1 == v2 || v1 < 0 || v2 < 0 || v1 >= nofVertices || v2 >= nofVertices)
		return false; // avoid loops and indices out of range
	return adjMatrix[getAdjIndex(nofVertices, v1, v2)];
}

int Graph::getDegree(int v) const
{
	if (v < 0 || v >= nofVertices) return -1;
	return degrees[v];
}

Graph* genRandomGraph(int nofVertices, double edgeDensity, int seed)
{
	std::minstd_rand rnd_generator(seed);
	int nofEdges = static_cast<int>(ceil((edgeDensity * nofVertices * (nofVertices - 1)) / 2.0));

	Graph* G = new Graph(nofVertices);

	int ec = 0;
	while (ec < nofEdges) {
		int v1 = rnd_generator() % nofVertices;
		int v2 = rnd_generator() % nofVertices;
		if (G->addEdge(v1, v2)) {
			ec++;
		}
	}

	return G;
}

static Graph* genPermutedGraph(const Graph& baseG, const int perm[])
{
	const int nofNodes = baseG.getNofVertices();
	Graph* G = new Graph(nofNodes);
	const bool* adjMatrix = baseG.getAdjMatrix();
	int pos = 0;
	for (int v1 = 0; v1 < nofNodes - 1; v1++) {
		for (int v2 = v1 + 1; v2 < nofNodes; v2++) {
			if (adjMatrix[pos]) {
				G->addEdge(perm[v1], perm[v2]);
			}
			pos++;
		}
	}
	return G;
}

Graph* genIsomorphicGraph(const Graph& baseG, int seed)
{
	std::default_random_engine rnd_generator(seed);
	const int nofNodes = baseG.getNofVertices();

	int* perm = new int[nofNodes];
	for (int i = 0; i < nofNodes; i++) {
		perm[i] = i;
	}
	std::shuffle(perm, perm + nofNodes, rnd_generator);
	Graph* G = genPermutedGraph(baseG, perm);
	delete[] perm;
	return G;
}

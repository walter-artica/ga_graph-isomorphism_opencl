#ifndef _GRAPH_H

#define _GRAPH_H

class Graph
{
private:
	int nofVertices;
	int nofEdges;
	int* degrees;
	bool* adjMatrix;
public:
	Graph(int nofVertices);
	~Graph();
	bool addEdge(int v1, int v2);
	bool isEdge(int v1, int v2) const;
	int getDegree(int v) const;
	int getNofVertices() const { return nofVertices; }
	int getNofEdges() const { return nofEdges; }
	const int* getDegreeVector() const { return degrees; }
	const bool* getAdjMatrix() const { return adjMatrix; }
	int getAdjMatrixLen() const { return nofVertices*(nofVertices - 1) / 2; }
};

Graph* genRandomGraph(int nofVertices, double edgeDensity, int seed = 128);
Graph* genIsomorphicGraph(const Graph& baseG, int seed = 256);

#endif

#ifndef _GA_H

#define _GA_H

#ifdef __APPLE__
#	include <OpenCL/cl.h>
#else
#	include <CL/cl.h>
#endif
#include <vector>
#include "graph.h"

struct GA_Config
{
	int seed;
	int nofVertices;
	double edgeDensity;
	int popLen;
	int maxGens;
	int nofOffspring;
	float w1, w2;
	float mutationProbability;

	GA_Config()
	{
		seed = 128;
		nofVertices = 50;
		edgeDensity = 0.1;
		popLen = 100;
		maxGens = 5;
		nofOffspring = 20;
		w1 = 0.1f;
		w2 = 0.9f;
		mutationProbability = 0.2f;
	}
};

class GeneticAlgorithm
{
private:
	Graph *G1, *G2;
	GA_Config config;
	double elapsedTimeInSeconds;
	std::vector<float> bestFitnessPerGeneration;
	size_t totalUsedMemoryInBytes;

	void initOpenCL(cl_uint platformIndex, cl_uint deviceIndex);
	void initKernels();
	void initBuffers();

	cl_device_id device;
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;

	cl_kernel krnlGeneratePopulation;
	cl_kernel krnlComputeFitness;
	cl_kernel krnlSelectParents;
	cl_kernel krnlCrossover;
	cl_kernel krnlMutate;
	cl_kernel krnlLocalOptimizationStep0;
	cl_kernel krnlLocalOptimizationStep1;
	cl_kernel krnlLocalOptimizationStep2;
	cl_kernel krnlLocalOptimizationStep3;
	cl_kernel krnlLocalOptimizationStep4;
	cl_kernel krnlGetIndicesByFitness;
	cl_kernel krnlRefreshPopulation;
	cl_kernel krnlGetBestFitness;

	cl_mem population_chromo_dev;
	cl_mem population_chromo_rev_dev;
	cl_mem population_fit_dev;
	cl_mem children_chromo_dev;
	cl_mem children_chromo_rev_dev;
	cl_mem children_fit_dev;
	cl_mem selected_indices_dev;
	cl_mem G1_degs_dev, G2_degs_dev;
	cl_mem G1_adjmat_dev, G2_adjmat_dev;
	cl_mem temp_optimization_results_dev;
	cl_mem indices_by_fit_dev;
	cl_mem best_fit_dev;
	cl_int randomSeed;

	void generatePopulation();
	void generateChildren();
	void refreshPopulation();
	float getBestFitness();
public:
	GeneticAlgorithm(cl_uint platformIndex, cl_uint deviceIndex);
	GeneticAlgorithm(const GA_Config& config, cl_uint platformIndex, cl_uint deviceIndex);
	~GeneticAlgorithm();
	void execute();
	int getTotalGenerations();
	double getElapsedTimeInSeconds();
	const std::vector<float>& getBestFitnessPerGeneration();
	size_t getTotalUsedMemory();
};

#endif

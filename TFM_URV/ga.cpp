#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <cstring>
#include <cassert>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <malloc.h>
#include "ga.h"
#include "opencl_helpers.h"

const char* getDefinitionsForCl(const GA_Config& config, int adjMatrixLen);
void dumpPopulation(const GA_Config config,
					const cl_command_queue cmd_queue,
					const cl_mem pop_chromo_dev, const cl_mem pop_fit_dev,
					const char* filename);
void dumpChildren(const GA_Config config,
				  const cl_command_queue cmd_queue,
				  const cl_mem children_chromo_dev, const cl_mem children_fit_dev,
				  const char* filename);

/**************************************************************************************************
** OpenCL helper routines
**************************************************************************************************/

const char* getDefinitionsForCl(const GA_Config& config, int adjMatrixLen)
{
	std::stringstream ss;
	ss << "#define POPULATION_LEN\t" << config.popLen << std::endl;
	ss << "#define CHROMOSOME_LEN\t" << config.nofVertices << std::endl;
	ss << "#define NOF_NODES\t" << config.nofVertices << std::endl;
	ss << "#define NOF_OFFSPRING\t" << config.nofOffspring << std::endl;
	ss << "#define ADJMATRIX_LEN\t" << adjMatrixLen << std::endl;
	ss << "#define W1\t" << config.w1 << "f" << std::endl;
	ss << "#define W2\t" << config.w2 << "f" << std::endl;
	ss << "#define MUTATION_PROBABILITY\t" << config.mutationProbability << "f" << std::endl;
	std::string s = ss.str();
	char *ret = new char[s.length() + 1];
	strcpy(ret, s.c_str());
	return ret;
}

bool checkUniqueness(const int* v, size_t len)
{
	std::vector<int> x(v, v + len);
	std::sort(x.begin(), x.end());
	return std::unique(x.begin(), x.end()) == x.end();
}

void dumpPopulation(const GA_Config config,
					const cl_command_queue cmd_queue,
					const cl_mem pop_chromo_dev, const cl_mem pop_fit_dev,
					const char* filename)
{
	cl_int* pop_chromo = new cl_int[config.nofVertices * config.popLen];
	cl_float* pop_fit = new cl_float[config.popLen];
	CHECK_FOR_ERROR(
		clEnqueueReadBuffer(cmd_queue, pop_chromo_dev, CL_TRUE, 0,
							sizeof(cl_int) * config.nofVertices * config.popLen, pop_chromo, 0, NULL, NULL)
	);
	CHECK_FOR_ERROR(
		clEnqueueReadBuffer(cmd_queue, pop_fit_dev, CL_TRUE, 0,
							sizeof(cl_float) * config.popLen, pop_fit, 0, NULL, NULL)
	);
	FILE* fp = fopen(filename, "wt");
	assert(fp != NULL);
	for (int i = 0; i < config.popLen; i++) {
		cl_int* c = &pop_chromo[i * config.nofVertices];
		if (!checkUniqueness(c, config.nofVertices))
			fprintf(fp, "**%02d**\n", i);
		else
			fprintf(fp, "==%02d==\n", i);
		for (int j = 0; j < config.nofVertices; j++) {
			fprintf(fp, "%02d ", c[j]);
		}
		fprintf(fp, "\n(%f)\n", pop_fit[i]);
	}
	fclose(fp);
	delete[] pop_chromo, pop_fit;
}

void dumpChildren(const GA_Config config,
				  const cl_command_queue cmd_queue,
				  const cl_mem children_chromo_dev, const cl_mem children_fit_dev,
				  const char* filename)
{
	cl_int* children_chromo = new cl_int[config.nofVertices * config.nofOffspring];
	cl_float* children_fit = new cl_float[config.nofOffspring];
	CHECK_FOR_ERROR(
		clEnqueueReadBuffer(cmd_queue, children_chromo_dev, CL_TRUE, 0,
							sizeof(cl_int) * config.nofVertices * config.nofOffspring,
							children_chromo,
							0, NULL, NULL)
	);
	CHECK_FOR_ERROR(
		clEnqueueReadBuffer(cmd_queue, children_fit_dev, CL_TRUE, 0,
							sizeof(cl_float) * config.nofOffspring,
							children_fit,
							0, NULL, NULL)
	);
	FILE* fp = fopen(filename, "wt");
	assert(fp != NULL);
	for (int i = 0; i < config.nofOffspring; i++) {
		int* ind = &children_chromo[i * config.nofVertices];
		if (!checkUniqueness(ind, config.nofVertices))
			fprintf(fp, "**%02d**\n", i);
		else
			fprintf(fp, "==%02d==\n", i);
		for (int j = 0; j < config.nofVertices; j++) {
			fprintf(fp, "%02d ", ind[j]);
		}
		fprintf(fp, "\n(%f)\n", children_fit[i]);
	}
	fclose(fp);
	delete[] children_chromo, children_fit;
}

/**************************************************************************************************
** GeneticAlgorithm methods
**************************************************************************************************/

GeneticAlgorithm::GeneticAlgorithm(cl_uint platformIndex, cl_uint deviceIndex)
	: GeneticAlgorithm(GA_Config(), platformIndex, deviceIndex)
{ }

GeneticAlgorithm::GeneticAlgorithm(const GA_Config& config, cl_uint platformIndex, cl_uint deviceIndex)
{
	this->config = config;
	this->randomSeed = config.seed;

	// Generate a pair of isomorphic graphs
	G1 = genRandomGraph(config.nofVertices, config.edgeDensity);
	G2 = genIsomorphicGraph(*G1);

	initOpenCL(platformIndex, deviceIndex);
	initKernels();
	initBuffers();
}

GeneticAlgorithm::~GeneticAlgorithm()
{
	CHECK_FOR_ERROR(clReleaseMemObject(population_chromo_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(population_chromo_rev_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(population_fit_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(children_chromo_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(children_chromo_rev_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(children_fit_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(selected_indices_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(G1_degs_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(G2_degs_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(G1_adjmat_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(G2_adjmat_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(temp_optimization_results_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(indices_by_fit_dev));
	CHECK_FOR_ERROR(clReleaseMemObject(best_fit_dev));

	CHECK_FOR_ERROR(clReleaseKernel(krnlGeneratePopulation));
	CHECK_FOR_ERROR(clReleaseKernel(krnlComputeFitness));
	CHECK_FOR_ERROR(clReleaseKernel(krnlSelectParents));
	CHECK_FOR_ERROR(clReleaseKernel(krnlCrossover));
	CHECK_FOR_ERROR(clReleaseKernel(krnlMutate));
	CHECK_FOR_ERROR(clReleaseKernel(krnlLocalOptimizationStep0));
	CHECK_FOR_ERROR(clReleaseKernel(krnlLocalOptimizationStep1));
	CHECK_FOR_ERROR(clReleaseKernel(krnlLocalOptimizationStep2));
	CHECK_FOR_ERROR(clReleaseKernel(krnlLocalOptimizationStep3));
	CHECK_FOR_ERROR(clReleaseKernel(krnlLocalOptimizationStep4));
	CHECK_FOR_ERROR(clReleaseKernel(krnlGetIndicesByFitness));
	CHECK_FOR_ERROR(clReleaseKernel(krnlRefreshPopulation));
	CHECK_FOR_ERROR(clReleaseKernel(krnlGetBestFitness));

	CHECK_FOR_ERROR(clReleaseProgram(program));
	CHECK_FOR_ERROR(clReleaseCommandQueue(cmd_queue));
	CHECK_FOR_ERROR(clReleaseContext(context));
	//clReleaseDevice(device);

	delete G1;
	delete G2;
}

int GeneticAlgorithm::getTotalGenerations()
{
	return static_cast<int>(bestFitnessPerGeneration.size());
}

void GeneticAlgorithm::execute()
{
	auto start_time = std::chrono::steady_clock::now();

	// Generate the initial population and calculate their fitnesses
	generatePopulation();
	// Try for maxGens, but exit prematurely if optimal solution is found
	for (int genInx = 0; genInx < config.maxGens; genInx++) {
		// Generate NOF_OFFSPRING individuals
		generateChildren();
		// Replace worst chromosomes with the new ones
		refreshPopulation();
		// Report results
		float best_fit = getBestFitness();
		printf("gen = %d, fit = %f\n", genInx, best_fit);
		bestFitnessPerGeneration.push_back(best_fit);
		// Exit if optimal solution is found
		if (best_fit <= 0.0) break;
	}
	clFinish(cmd_queue);

	auto end_time = std::chrono::steady_clock::now();
	auto elapsed = end_time - start_time;
	elapsedTimeInSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
}

double GeneticAlgorithm::getElapsedTimeInSeconds()
{
	return elapsedTimeInSeconds;
}

const std::vector<float>& GeneticAlgorithm::getBestFitnessPerGeneration()
{
	return bestFitnessPerGeneration;
}

size_t GeneticAlgorithm::getTotalUsedMemory()
{
	return totalUsedMemoryInBytes;
}

void GeneticAlgorithm::initOpenCL(cl_uint platformIndex, cl_uint deviceIndex)
{
	cl_int errcode;

	cl_uint num_platforms;
	CHECK_FOR_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));
	cl_platform_id* platforms = (cl_platform_id*)alloca(num_platforms*sizeof(cl_platform_id));
	CHECK_FOR_ERROR(clGetPlatformIDs(num_platforms, platforms, NULL));

	assert(0 <= platformIndex && platformIndex < num_platforms);
	const cl_platform_id& platform = platforms[platformIndex];

	cl_uint num_devices;
	CHECK_FOR_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
	cl_device_id* devices = (cl_device_id*)alloca(num_devices*sizeof(cl_device_id));
	CHECK_FOR_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

	assert(0 <= deviceIndex && deviceIndex < num_devices);
	this->device = devices[deviceIndex];

	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0
	};

	context = clCreateContext(properties, 1, &device, NULL, NULL, &errcode); CHECK_FOR_ERROR(errcode);

	cmd_queue = clCreateCommandQueue(context, device, 0, &errcode); CHECK_FOR_ERROR(errcode);

	const char *inctext = getDefinitionsForCl(config, G1->getAdjMatrixLen());
	const char *srctext = readFileIntoString("kernel.cl");
	const char *strings[] = { inctext, srctext };
	program = clCreateProgramWithSource(context, 2, strings, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	errcode = clBuildProgram(program, 0, NULL, "-Werror -DNDEBUG", NULL, NULL);
	checkBuildProgram(errcode, program, device);

	delete[] srctext, inctext;
}

void GeneticAlgorithm::initKernels()
{
	cl_int errcode;

	krnlGeneratePopulation = clCreateKernel(program, "generatePopulation", &errcode); CHECK_FOR_ERROR(errcode);
	krnlComputeFitness = clCreateKernel(program, "computeFitness", &errcode); CHECK_FOR_ERROR(errcode);
	krnlSelectParents = clCreateKernel(program, "selectParents", &errcode); CHECK_FOR_ERROR(errcode);
	krnlCrossover = clCreateKernel(program, "crossover", &errcode); CHECK_FOR_ERROR(errcode);
	krnlMutate = clCreateKernel(program, "mutate", &errcode); CHECK_FOR_ERROR(errcode);
	krnlLocalOptimizationStep0 = clCreateKernel(program, "localOptimization_step0", &errcode); CHECK_FOR_ERROR(errcode);
	krnlLocalOptimizationStep1 = clCreateKernel(program, "localOptimization_step1", &errcode); CHECK_FOR_ERROR(errcode);
	krnlLocalOptimizationStep2 = clCreateKernel(program, "localOptimization_step2", &errcode); CHECK_FOR_ERROR(errcode);
	krnlLocalOptimizationStep3 = clCreateKernel(program, "localOptimization_step3", &errcode); CHECK_FOR_ERROR(errcode);
	krnlLocalOptimizationStep4 = clCreateKernel(program, "localOptimization_step4", &errcode); CHECK_FOR_ERROR(errcode);
	krnlGetIndicesByFitness = clCreateKernel(program, "getIndicesByFitness", &errcode); CHECK_FOR_ERROR(errcode);
	krnlRefreshPopulation = clCreateKernel(program, "refreshPopulation", &errcode); CHECK_FOR_ERROR(errcode);
	krnlGetBestFitness = clCreateKernel(program, "getBestFitness", &errcode); CHECK_FOR_ERROR(errcode);
}

void GeneticAlgorithm::initBuffers()
{
	cl_int errcode;

	size_t stdChromoSize = sizeof(cl_int) * config.nofVertices;
	size_t popChromoSize = stdChromoSize * config.popLen;
	size_t popFitSize = sizeof(cl_float) * config.popLen;
	size_t selectedIndicesSize = sizeof(cl_int) * config.nofOffspring * 2;
	size_t childrenChromoSize = stdChromoSize * config.nofOffspring;
	size_t childrenFitSize = sizeof(cl_float) * config.nofOffspring;
	size_t revBufSize = stdChromoSize * config.popLen;
	size_t tempOptimizationResultsSize = sizeof(cl_int3) * config.nofVertices * config.nofOffspring;
	size_t indicesByFitSize = sizeof(cl_int) * config.popLen;
	size_t bestFitSize = sizeof(cl_float);
	size_t degreeVectorSize = sizeof(cl_int) * config.nofVertices;
	size_t adjMatrixSize = sizeof(cl_char) * G1->getAdjMatrixLen();

	population_chromo_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, popChromoSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	population_chromo_rev_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, popChromoSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	population_fit_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, popFitSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	selected_indices_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, selectedIndicesSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	children_chromo_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, childrenChromoSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	children_chromo_rev_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, childrenChromoSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	children_fit_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, childrenFitSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	temp_optimization_results_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, tempOptimizationResultsSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	indices_by_fit_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, indicesByFitSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	best_fit_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bestFitSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	G1_degs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, degreeVectorSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	G2_degs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, degreeVectorSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	G1_adjmat_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, adjMatrixSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);
	G2_adjmat_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, adjMatrixSize, NULL, &errcode); CHECK_FOR_ERROR(errcode);

	size_t totalBytes = popChromoSize*2 + popFitSize + selectedIndicesSize + childrenChromoSize*2 + childrenFitSize
		+ revBufSize + tempOptimizationResultsSize + indicesByFitSize + bestFitSize + degreeVectorSize * 2 + adjMatrixSize * 2;
	this->totalUsedMemoryInBytes = totalBytes;

	CHECK_FOR_ERROR(clEnqueueWriteBuffer(cmd_queue, G1_degs_dev, CL_TRUE, 0, degreeVectorSize, G1->getDegreeVector(), 0, NULL, NULL));
	CHECK_FOR_ERROR(clEnqueueWriteBuffer(cmd_queue, G2_degs_dev, CL_TRUE, 0, degreeVectorSize, G2->getDegreeVector(), 0, NULL, NULL));
	CHECK_FOR_ERROR(clEnqueueWriteBuffer(cmd_queue, G1_adjmat_dev, CL_TRUE, 0, adjMatrixSize, G1->getAdjMatrix(), 0, NULL, NULL));
	CHECK_FOR_ERROR(clEnqueueWriteBuffer(cmd_queue, G2_adjmat_dev, CL_TRUE, 0, adjMatrixSize, G2->getAdjMatrix(), 0, NULL, NULL));
}

void GeneticAlgorithm::generatePopulation()
{
	size_t globalws[1] = { (size_t)config.popLen };

	CHECK_FOR_ERROR(clSetKernelArg(krnlGeneratePopulation, 0, sizeof(cl_mem), &population_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlGeneratePopulation, 1, sizeof(cl_mem), &population_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlGeneratePopulation, 2, sizeof(cl_int), &randomSeed));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlGeneratePopulation, 1, NULL, globalws, NULL, 0, NULL, NULL));
	randomSeed += config.popLen;

	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 0, sizeof(cl_mem), &population_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 1, sizeof(cl_mem), &population_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 2, sizeof(cl_mem), &population_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 3, sizeof(cl_mem), &G1_degs_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 4, sizeof(cl_mem), &G1_adjmat_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 5, sizeof(cl_mem), &G2_degs_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlComputeFitness, 6, sizeof(cl_mem), &G2_adjmat_dev));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlComputeFitness, 1, NULL, globalws, NULL, 0, NULL, NULL));

	//dumpPopulation(config, cmd_queue, population_chromo_dev, population_fit_dev, "out/initialPopDump.txt");
}

void GeneticAlgorithm::generateChildren()
{
	const size_t globalws[1] = { (size_t)config.nofOffspring };
	const size_t globalws_f1[2] = { (size_t)config.nofOffspring, (size_t)config.nofVertices - 1 };
	const size_t globalws_f2[2] = { (size_t)config.nofOffspring, (size_t)config.nofVertices };

	CHECK_FOR_ERROR(clSetKernelArg(krnlSelectParents, 0, sizeof(cl_mem), &selected_indices_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlSelectParents, 1, sizeof(cl_mem), &population_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlSelectParents, 2, sizeof(cl_int), &randomSeed));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlSelectParents, 1, NULL, globalws, NULL, 0, NULL, NULL));
	randomSeed += config.popLen;

	CHECK_FOR_ERROR(clSetKernelArg(krnlCrossover, 0, sizeof(cl_mem), &children_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlCrossover, 1, sizeof(cl_mem), &children_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlCrossover, 2, sizeof(cl_mem), &population_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlCrossover, 3, sizeof(cl_mem), &selected_indices_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlCrossover, 4, sizeof(cl_int), &randomSeed));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlCrossover, 1, NULL, globalws, NULL, 0, NULL, NULL));
	randomSeed += config.popLen;

	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 0, sizeof(cl_mem), &children_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 1, sizeof(cl_mem), &children_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 2, sizeof(cl_mem), &children_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 3, sizeof(cl_mem), &G1_degs_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 4, sizeof(cl_mem), &G1_adjmat_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 5, sizeof(cl_mem), &G2_degs_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 6, sizeof(cl_mem), &G2_adjmat_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlMutate, 7, sizeof(cl_int), &randomSeed));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlMutate, 1, NULL, globalws, NULL, 0, NULL, NULL));
	randomSeed += config.popLen;

	for (cl_int v1 = 0; v1 < config.nofVertices; v1++) {
		for (cl_int v2 = 0; v2 < config.nofVertices; v2++) {
			if (v1 != v2) {
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep0, 0, sizeof(cl_mem), &children_chromo_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep0, 1, sizeof(cl_mem), &children_chromo_rev_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep0, 2, sizeof(cl_int), &v1));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep0, 3, sizeof(cl_int), &v2));
				CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlLocalOptimizationStep0, 1, NULL, globalws, NULL, 0, NULL, NULL));

				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep1, 0, sizeof(cl_mem), &temp_optimization_results_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep1, 1, sizeof(cl_mem), &children_chromo_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep1, 2, sizeof(cl_mem), &G1_adjmat_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep1, 3, sizeof(cl_mem), &G2_adjmat_dev));
				CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlLocalOptimizationStep1, 2, NULL, globalws_f1, NULL, 0, NULL, NULL));

				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep2, 0, sizeof(cl_mem), &temp_optimization_results_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep2, 1, sizeof(cl_mem), &children_chromo_rev_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep2, 2, sizeof(cl_mem), &G1_adjmat_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep2, 3, sizeof(cl_mem), &G2_adjmat_dev));
				CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlLocalOptimizationStep2, 2, NULL, globalws_f1, NULL, 0, NULL, NULL));

				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep3, 0, sizeof(cl_mem), &temp_optimization_results_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep3, 1, sizeof(cl_mem), &children_chromo_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep3, 2, sizeof(cl_mem), &G1_degs_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep3, 3, sizeof(cl_mem), &G2_degs_dev));
				CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlLocalOptimizationStep3, 2, NULL, globalws_f2, NULL, 0, NULL, NULL));

				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 0, sizeof(cl_mem), &children_chromo_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 1, sizeof(cl_mem), &children_chromo_rev_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 2, sizeof(cl_mem), &children_fit_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 3, sizeof(cl_mem), &temp_optimization_results_dev));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 4, sizeof(cl_int), &v1));
				CHECK_FOR_ERROR(clSetKernelArg(krnlLocalOptimizationStep4, 5, sizeof(cl_int), &v2));
				CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlLocalOptimizationStep4, 1, NULL, globalws, NULL, 0, NULL, NULL));
			}
		}
	}

	//dumpChildren(config, cmd_queue, children_chromo_dev, children_fit_dev, "out/new_children.txt");
}

void GeneticAlgorithm::refreshPopulation()
{
	size_t globalws[1] = { 1 };
	CHECK_FOR_ERROR(clSetKernelArg(krnlGetIndicesByFitness, 0, sizeof(cl_mem), &indices_by_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlGetIndicesByFitness, 1, sizeof(cl_mem), &population_fit_dev));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlGetIndicesByFitness, 1, NULL, globalws, NULL, 0, NULL, NULL));

	//cl_int* indbyfit = new cl_int[config.popLen];
	//CHECK_FOR_ERROR(
	//	clEnqueueReadBuffer(cmd_queue, indices_by_fit_dev, CL_TRUE, 0,
	//	sizeof(cl_int) * config.popLen, indbyfit, 0, NULL, NULL)
	//	);
	//assert(checkUniqueness(indbyfit, config.popLen));
	//for (int i = 0; i < config.popLen; i++) {
	//	printf("%d  ", indbyfit[i]);
	//}
	//printf("\n");
	//delete[] indbyfit;

	globalws[0] = (size_t) config.nofOffspring;
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 0, sizeof(cl_mem), &population_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 1, sizeof(cl_mem), &population_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 2, sizeof(cl_mem), &population_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 3, sizeof(cl_mem), &children_chromo_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 4, sizeof(cl_mem), &children_chromo_rev_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 5, sizeof(cl_mem), &children_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlRefreshPopulation, 6, sizeof(cl_mem), &indices_by_fit_dev));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlRefreshPopulation, 1, NULL, globalws, NULL, 0, NULL, NULL));
}

float GeneticAlgorithm::getBestFitness()
{
	size_t globalws[1] = { 1 };
	cl_float best_fit;

	CHECK_FOR_ERROR(clSetKernelArg(krnlGetBestFitness, 0, sizeof(cl_mem), &best_fit_dev));
	CHECK_FOR_ERROR(clSetKernelArg(krnlGetBestFitness, 1, sizeof(cl_mem), &population_fit_dev));
	CHECK_FOR_ERROR(clEnqueueNDRangeKernel(cmd_queue, krnlGetBestFitness, 1, NULL, globalws, NULL, 0, NULL, NULL));
	CHECK_FOR_ERROR(clEnqueueReadBuffer(cmd_queue, best_fit_dev, CL_TRUE, 0, sizeof(cl_float), &best_fit, 0, NULL, NULL));

	return best_fit;
}

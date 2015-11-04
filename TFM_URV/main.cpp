#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "ga.h"

const int PLATFORM_INDEX = 0;
const int DEVICE_INDEX = 0;

int main(int argc, char *argv[])
{
	int seed = 256;
	if (argc > 1) {
		seed = atoi(argv[1]);
	}
	printf("Seed = %d\n\n", seed);

	std::vector<int> node_count{ 30, 50, 70, 90, 110 };
	std::vector<double> edge_density{ 0.1, 0.2 };

	char filename[256];
	unsigned dummy_timestamp = static_cast<unsigned>(time(NULL));
	sprintf(filename, "out/results-%d-%u.csv", seed, dummy_timestamp);
	FILE* report = fopen(filename, "wt");

	GA_Config config;
	config.seed = seed;

	for (auto nc : node_count) {
		for (auto ed : edge_density) {
			printf("nc = %d, ed = %.2f\n", nc, ed);
			config.nofVertices = nc;
			config.edgeDensity = ed;
			GeneticAlgorithm ga(config, PLATFORM_INDEX, DEVICE_INDEX);
			ga.execute();
			float bestFit = ga.getBestFitnessPerGeneration().back();
			size_t totalUsedMemory = ga.getTotalUsedMemory();
#ifdef _MSC_VER
			fprintf(report, "%d, %f, %f, %d, %f, %Iu\n", nc, ed, ga.getElapsedTimeInSeconds(), ga.getTotalGenerations(), bestFit, totalUsedMemory);
#else
			fprintf(report, "%d, %f, %f, %d, %f, %zu\n", nc, ed, ga.getElapsedTimeInSeconds(), ga.getTotalGenerations(), bestFit, totalUsedMemory);
#endif
		}
	}

	fclose(report);

	return 0;
}

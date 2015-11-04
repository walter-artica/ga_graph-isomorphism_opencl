#line 2
///////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
///////////////////////////////////////////////////////////////////////////////////////////////////

// Fix for AMD's printf bug
#define printf(fmt, ...) printf((constant char*)fmt, ##__VA_ARGS__)

#ifndef NDEBUG
#	define assert(x)		if (!(x)) {	printf("Assert(%s) at line %d\n", #x, __LINE__); }
#	define assert_msg(x,s)	if (!(x)) {	printf("Assert(%s) at line %d\n", s, __LINE__); }
#else
#	define assert(x)
#	define assert_msg(x,s)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////////////////////////

typedef char BOOL;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
///////////////////////////////////////////////////////////////////////////////////////////////////

int rand_int(int *seed);
float rand_float(int *seed);
void swapIntsInArray(global int* x, int i, int j);
void generateIndividual(global int* chromo, global int* chromo_rev, int* random_seed);
int getFitness_f1_1(global const int* perm,
					global const BOOL* G1_adjmat, global const BOOL* G2_adjmat,
					const int v1);
int getFitness_f1_2(global const int* rev_perm,
					global const BOOL* G1_adjmat, global const BOOL* G2_adjmat,
					const int v1);
int getFitness_f2(global const int* perm,
				  global const int* G1_degs, global const int* G2_degs,
				  const int v);
int getAdjIndex(int v1, int v2);
int isEdge(global const BOOL* adjMatrix, int v1, int v2);
int getTournamentWinner(global const float* fitnesses, int* random_seed);
void doSelectParents(int* par1, int* par2, global const float* fitnesses, int* random_seed);
void doCrossover(global int* newChild_chromo,
				 global int* newChild_chromo_rev,
				 global const int* par1, global const int* par2,
				 int crossLocation, int inx);
void doMutation(global int* child_chromo, global int* child_chromo_rev, int* random_seed);
float getFitnessInOneStep(global const int* perm,
						  global const int* rev_perm,
						  global const int* G1_degs, global const BOOL* G1_adjmat,
						  global const int* G2_degs, global const BOOL* G2_adjmat);

///////////////////////////////////////////////////////////////////////////////////////////////////
//  Helper functions
///////////////////////////////////////////////////////////////////////////////////////////////////

#define A	16807
#define	M	2147483647
#define	Q	(M / A)
#define	R	(M % A)

int rand_int(int *seed)
{
	int gamma = A*(*seed % Q) - R*(*seed / Q);
	if (gamma > 0)
		*seed = gamma;
	else
		*seed = gamma + M;
	return *seed - 1;
}

float rand_float(int *seed)
{
	return rand_int(seed) * (1.0f / M);
}

void swapIntsInArray(__global int* x, int i, int j)
{
	int tmp = x[i];
	x[i] = x[j];
	x[j] = tmp;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GA functions
///////////////////////////////////////////////////////////////////////////////////////////////////

void generateIndividual(global int* chromo, global int* chromo_rev, int* random_seed)
{
	for (int i = 0; i < CHROMOSOME_LEN; i++) {
		chromo[i] = i;
		chromo_rev[i] = i;
	}
	for (int i = 0; i < CHROMOSOME_LEN; i++) {
		int j = rand_int(random_seed) % CHROMOSOME_LEN;
		swapIntsInArray(chromo, i, j);
		swapIntsInArray(chromo_rev, chromo[i], chromo[j]);
	}
}

int getAdjIndex(int v1, int v2)
{
	if (v1 > v2) {
		int tmp = v1;
		v1 = v2;
		v2 = tmp;
	}
	return v1*NOF_NODES + v2 - (v1 + 1)*(v1 + 2) / 2;
}

int isEdge(global const BOOL* adjMatrix, int v1, int v2)
{
	int linearInx = getAdjIndex(v1, v2);
	assert_msg(0 <= linearInx && linearInx < ADJMATRIX_LEN, "isEdge invariant");
	return adjMatrix[linearInx] != 0;
}

float getFitnessInOneStep(global const int* perm,
						  global const int* rev_perm,
						  global const int* G1_degs, global const BOOL* G1_adjmat,
						  global const int* G2_degs, global const BOOL* G2_adjmat)
{
	int pos;
	global const int* degrees1 = G1_degs;
	global const int* degrees2 = G2_degs;
	global const BOOL* adjMatrix1 = G1_adjmat;
	global const BOOL* adjMatrix2 = G2_adjmat;

	int f1_1 = 0;
	pos = 0;
	for (int v1 = 0; v1 < NOF_NODES - 1; v1++) {
		for (int v2 = v1 + 1; v2 < NOF_NODES; v2++) {
			if (adjMatrix1[pos]) {
				int vsrc = perm[v1];
				int vdst = perm[v2];
				assert_msg(0 <= vsrc && vsrc < NOF_NODES && 0 <= vdst && vdst < NOF_NODES, "fitness_initial part1");
				f1_1 += (!isEdge(adjMatrix2, vsrc, vdst)) ? 1 : 0;
			}
			pos++;
		}
	}
	int f1_2 = 0;
	pos = 0;
	for (int v1 = 0; v1 < NOF_NODES - 1; v1++) {
		for (int v2 = v1 + 1; v2 < NOF_NODES; v2++) {
			if (adjMatrix2[pos]) {
				int vsrc = rev_perm[v1];
				int vdst = rev_perm[v2];
				assert_msg(0 <= vsrc && vsrc < NOF_NODES && 0 <= vdst && vdst < NOF_NODES, "fitness_initial part2");
				f1_2 += (!isEdge(adjMatrix1, vsrc, vdst)) ? 1 : 0;
			}
			pos++;
		}
	}
	int f1 = f1_1 + f1_2;

	int f2 = 0;
	for (int v = 0; v < NOF_NODES; v++) {
		int d1 = degrees1[v];
		int p = perm[v];
		assert_msg(0 <= p && p < NOF_NODES, "fitness_initial part3");
		int d2 = degrees2[p];
		f2 += (d1 != d2) ? 1 : 0;
	}

	return W1*f1 + W2*f2;
}

int getFitness_f1_1(global const int* perm,
					global const BOOL* G1_adjmat, global const BOOL* G2_adjmat,
					const int v1)
{
	int f1 = 0;
	int pos = getAdjIndex(v1, v1+1);
	for (int v2 = v1 + 1; v2 < NOF_NODES; v2++) {
		if (G1_adjmat[pos] != 0) {
			int vsrc = perm[v1];
			int vdst = perm[v2];
			assert_msg(0 <= vsrc && vsrc < NOF_NODES && 0 <= vdst && vdst < NOF_NODES, "fitness part1");
			f1 += isEdge(G2_adjmat, vsrc, vdst) ? 0 : 1;
		}
		pos++;
	}
	
	return f1;
}

int getFitness_f1_2(global const int* rev_perm,
					global const BOOL* G1_adjmat, global const BOOL* G2_adjmat,
					const int v1)
{
	int f1 = 0;
	int pos = getAdjIndex(v1, v1 + 1);
	for (int v2 = v1 + 1; v2 < NOF_NODES; v2++) {
		if (G2_adjmat[pos] != 0) {
			int vsrc = rev_perm[v1];
			int vdst = rev_perm[v2];
			assert_msg(0 <= vsrc && vsrc < NOF_NODES && 0 <= vdst && vdst < NOF_NODES, "fitness part2");
			f1 += isEdge(G1_adjmat, vsrc, vdst) ? 0 : 1;
		}
		pos++;
	}

	return f1;
}

int getFitness_f2(global const int* perm,
				  global const int* G1_degs, global const int* G2_degs,
				  const int v)
{
	int d1 = G1_degs[v];
	int p = perm[v];
	assert_msg(0 <= p && p < NOF_NODES, "fitness part3");
	int d2 = G2_degs[p];
	assert_msg(d1 >= 0 && d2 >= 0, "fitness: negative degrees");
	int f2 = (int)(d1 != d2);

	return f2;
}

int getTournamentWinner(global const float* fitnesses, int* random_seed)
{
	int win1 = rand_int(random_seed) % POPULATION_LEN;
	int win2 = rand_int(random_seed) % POPULATION_LEN;
	assert(0 <= win1 && win1 < POPULATION_LEN && 0 <= win2 && win2 < POPULATION_LEN);
	return (fitnesses[win1] < fitnesses[win2]) ? win1 : win2;
}

inline void doSelectParents(int* par1, int* par2, global const float* fitnesses, int* random_seed)
{
	*par1 = getTournamentWinner(fitnesses, random_seed);
	*par2 = getTournamentWinner(fitnesses, random_seed);
}

void genReversePermutation(global int* rev_perm, global const int* perm)
{
	for (int i = 0; i < CHROMOSOME_LEN; i++) {
		int j = 0;
		while (perm[j] != i)
			j++;
		rev_perm[i] = j;
	}
}

void doCrossover(global int* newChild_chromo,
				 global int* newChild_chromo_rev,
				 global const int* par1, global const int* par2,
				 int crossLocation, int inx)
{
	assert_msg(0 <= crossLocation && crossLocation < CHROMOSOME_LEN, "cross-location out of bounds");

	for (int i = 0; i <= crossLocation; i++) {
		newChild_chromo[i] = par1[i];
	}

	const int remaining = CHROMOSOME_LEN - crossLocation - 1;
	int cnt = 0;
	for (int i = 0; i < CHROMOSOME_LEN; i++) { // Loop parent
		bool inChild = false;
		int candidate = par2[i];
		for (int j = 0; j <= crossLocation; j++) {	// Loop child
			if (newChild_chromo[j] == candidate) {
				inChild = true;
				break;
			}
		}
		if (!inChild) {
			cnt++;
			newChild_chromo[crossLocation + cnt] = candidate;
		}
		if (cnt == remaining) break;
	}

	genReversePermutation(newChild_chromo_rev, newChild_chromo);
}

void doMutation(global int* child_chromo, global int* child_chromo_rev, int* random_seed)
{
	int sel1, sel2;

	float rnd = rand_float(random_seed);
	if (rnd < MUTATION_PROBABILITY) {
		// Get 2 different random numbers in the range [0, CHROMOSOME_LEN)
		sel1 = rand_int(random_seed) % CHROMOSOME_LEN;
		do {
			sel2 = rand_int(random_seed) % CHROMOSOME_LEN;
		} while (sel1 == sel2);
		assert(0 <= sel1 && sel1 < CHROMOSOME_LEN && 0 <= sel2 && sel2 < CHROMOSOME_LEN);
		swapIntsInArray(child_chromo, sel1, sel2);
		swapIntsInArray(child_chromo_rev, child_chromo[sel1], child_chromo[sel2]);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels
///////////////////////////////////////////////////////////////////////////////////////////////////

kernel void generatePopulation(global int* newpop_chromosomes,
							   global int* newpop_chromosomes_rev,
							   const int random_seed)
{
	const size_t inx = get_global_id(0);
	assert(inx < POPULATION_LEN);
	int seed = random_seed + 30*inx;
	global int* chromo = &newpop_chromosomes[inx*CHROMOSOME_LEN];
	global int* chromo_rev = &newpop_chromosomes_rev[inx*CHROMOSOME_LEN];
	generateIndividual(chromo, chromo_rev, &seed);
}

kernel void computeFitness(global int* pop_chromosomes,
						   global int* pop_chromosomes_rev,
						   global float* pop_fitnesses,
						   global const int* G1_degs, global const BOOL* G1_adjmat,
						   global const int* G2_degs, global const BOOL* G2_adjmat)
{
	const size_t inx = get_global_id(0);
	assert(inx < POPULATION_LEN);
	global int* chromo = &pop_chromosomes[inx*CHROMOSOME_LEN];
	global int* chromo_rev = &pop_chromosomes_rev[inx*CHROMOSOME_LEN];
	pop_fitnesses[inx] = getFitnessInOneStep(chromo, chromo_rev, G1_degs, G1_adjmat, G2_degs, G2_adjmat);
}

kernel void selectParents(global int* selectedIndices,
						  global const float* pop_fitnesses,
						  const int random_seed)
{
	const size_t inx = get_global_id(0);
	assert(inx < NOF_OFFSPRING);
	int seed = random_seed + inx;
	int pinx1, pinx2;
	doSelectParents(&pinx1, &pinx2, pop_fitnesses, &seed);
	assert(0 <= pinx1 && pinx1 < POPULATION_LEN && 0 <= pinx2 && pinx2 < POPULATION_LEN);
	selectedIndices[inx * 2] = pinx1;
	selectedIndices[inx * 2 + 1] = pinx2;
}

kernel void crossover(global int* children_chromosomes,
					  global int* children_chromosomes_rev,
					  global const int* pop_chromosomes,
					  global const int* selectedIndices,
					  const int random_seed)
{
	const size_t inx = get_global_id(0);
	assert(inx < NOF_OFFSPRING);
	int seed = random_seed + inx;
	int pinx1 = selectedIndices[inx * 2];
	int pinx2 = selectedIndices[inx * 2 + 1];
	assert(0 <= pinx1 && pinx1 < POPULATION_LEN && 0 <= pinx2 && pinx2 < POPULATION_LEN);
	global const int* par1_chromo = &pop_chromosomes[pinx1*CHROMOSOME_LEN];
	global const int* par2_chromo = &pop_chromosomes[pinx2*CHROMOSOME_LEN];
	global int* child_chromo = &children_chromosomes[inx*CHROMOSOME_LEN];
	global int* child_chromo_rev = &children_chromosomes_rev[inx*CHROMOSOME_LEN];
	int crossLocation = rand_int(&seed) % CHROMOSOME_LEN;
	doCrossover(child_chromo, child_chromo_rev, par1_chromo, par2_chromo, crossLocation, inx);
	// Note: children fitnesses have random values up to this point;
	//       'mutate' kernel will put the correct values (after doing the actual mutation)
}

kernel void mutate(global int* children_chromosomes,
				   global int* children_chromosomes_rev,
				   global float* children_fitnesses,
				   global const int* G1_degs, global const BOOL* G1_adjmat,
				   global const int* G2_degs, global const BOOL* G2_adjmat,
				   const int random_seed)
{
	const size_t inx = get_global_id(0);
	assert(inx < NOF_OFFSPRING);
	int seed = random_seed + inx;
	global int* chromo = &children_chromosomes[inx*CHROMOSOME_LEN];
	global int* chromo_rev = &children_chromosomes_rev[inx*CHROMOSOME_LEN];
	doMutation(chromo, chromo_rev, &seed);
	children_fitnesses[inx] = getFitnessInOneStep(chromo, chromo_rev, G1_degs, G1_adjmat, G2_degs, G2_adjmat);
}

kernel void localOptimization_step0(global int* children_chromosomes,
									global int* children_chromosomes_rev,
									const int v1, const int v2)
{
	const size_t child_index = get_global_id(0);
	assert(child_index < NOF_OFFSPRING);
	global int* chromo = &children_chromosomes[child_index*CHROMOSOME_LEN];
	global int* chromo_rev = &children_chromosomes_rev[child_index*CHROMOSOME_LEN];
	swapIntsInArray(chromo, v1, v2);
	swapIntsInArray(chromo_rev, chromo[v1], chromo[v2]);
}

kernel void localOptimization_step1(global int3* temp_results,
									global const int* children_chromosomes,
									global const BOOL* G1_adjmat, global const BOOL* G2_adjmat)
{
	const size_t child_index = get_global_id(0);
	const size_t v1 = get_global_id(1);
	assert(child_index < NOF_OFFSPRING);
	assert(v1 < CHROMOSOME_LEN - 1);
	const size_t child_offset = child_index*CHROMOSOME_LEN;
	global const int* chromo = &children_chromosomes[child_offset];

	int partial_f1 = getFitness_f1_1(chromo, G1_adjmat, G2_adjmat, v1);
	assert_msg(partial_f1 >= 0, "localOptimization_step1");
	temp_results[child_offset + v1].x = partial_f1;
}

kernel void localOptimization_step2(global int3* temp_results,
									global const int* children_chromosomes_rev,
									global const BOOL* G1_adjmat, global const BOOL* G2_adjmat)
{
	const size_t child_index = get_global_id(0);
	const size_t v1 = get_global_id(1);
	assert(child_index < NOF_OFFSPRING);
	assert(v1 < CHROMOSOME_LEN-1);
	const size_t child_offset = child_index*CHROMOSOME_LEN;
	global const int* chromo_rev = &children_chromosomes_rev[child_offset];

	int partial_f1 = getFitness_f1_2(chromo_rev, G1_adjmat, G2_adjmat, v1);
	assert_msg(partial_f1 >= 0, "localOptimization_step1");
	temp_results[child_offset + v1].y = partial_f1;
}

kernel void localOptimization_step3(global int3* temp_results,
									global const int* children_chromosomes,
									global const int* G1_degs, global const int* G2_degs)
{
	const size_t child_index = get_global_id(0);
	const size_t v = get_global_id(1);
	assert(child_index < NOF_OFFSPRING);
	assert(v < CHROMOSOME_LEN);
	const size_t child_offset = child_index*CHROMOSOME_LEN;
	global const int* chromo = &children_chromosomes[child_offset];

	temp_results[child_offset + v].z = getFitness_f2(chromo, G1_degs, G2_degs, v);
}

kernel void localOptimization_step4(global int* children_chromosomes,
									global int* children_chromosomes_rev,
									global float* children_fitnesses,
									global const int3* temp_results,
									const int v1, const int v2)
{
	const size_t child_index = get_global_id(0);
	assert(child_index < NOF_OFFSPRING);
	global int* chromo = &children_chromosomes[child_index*CHROMOSOME_LEN];
	global int* chromo_rev = &children_chromosomes_rev[child_index*CHROMOSOME_LEN];

	const size_t child_offset = child_index*CHROMOSOME_LEN;
	int f1 = 0;
	int f2 = 0;
	for (size_t i = 0; i < CHROMOSOME_LEN - 1; i++) {
		f1 += temp_results[child_offset + i].x + temp_results[child_offset + i].y;
		f2 += temp_results[child_offset + i].z;
	}
	f2 += temp_results[child_offset + CHROMOSOME_LEN - 1].z;
	float new_fit = W1 * f1 + W2 * f2;
	if (new_fit <= children_fitnesses[child_index]) {
		children_fitnesses[child_index] = new_fit;
	} else {
		swapIntsInArray(chromo, v1, v2);
		swapIntsInArray(chromo_rev, chromo[v1], chromo[v2]);
	}
}

kernel void getIndicesByFitness(global int* indices_by_fit, global const float* pop_fitnesses)
{
	for (int i = 0; i < POPULATION_LEN; i++) {
		indices_by_fit[i] = i;
	}
	for (int i = 0; i < POPULATION_LEN - 1; i++) {
		int k = i;
		int x = pop_fitnesses[indices_by_fit[i]];
		for (int j = i + 1; j < POPULATION_LEN; j++) {
			if (pop_fitnesses[indices_by_fit[j]] > x) {
				k = j;
				x = pop_fitnesses[indices_by_fit[k]];
			}
		}
		swapIntsInArray(indices_by_fit, i, k);
	}
}

kernel void refreshPopulation(global int* pop_chromosomes,
							  global int* pop_chromosomes_rev,
							  global float* pop_fitnesses,
							  global const int* children_chromosomes,
							  global const int* children_chromosomes_rev,
							  global const float* children_fitnesses,
							  global const int* indices_by_fit)
{
	const size_t inx = get_global_id(0);
	assert(inx < NOF_OFFSPRING);
	size_t child_index = inx;
	size_t removed_index = indices_by_fit[inx];
	pop_fitnesses[removed_index] = children_fitnesses[child_index];
	const size_t basePop = removed_index*CHROMOSOME_LEN;
	const size_t baseChild = child_index*CHROMOSOME_LEN;
	for (int i = 0; i < CHROMOSOME_LEN; i++) {
		pop_chromosomes[basePop + i] = children_chromosomes[baseChild + i];
		pop_chromosomes_rev[basePop + i] = children_chromosomes_rev[baseChild + i];
	}
}

kernel void getBestFitness(global float* best_fit, global const float* pop_fitnesses)
{
	float best = INFINITY;

	for (int i = 0; i < POPULATION_LEN; i++) {
		float fit = pop_fitnesses[i];
		if (fit < best) {
			best = fit;
		}
	}
	*best_fit = best;
}

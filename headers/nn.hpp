#ifndef NN_GPU_NN_HPP
#define NN_GPU_NN_HPP

#include "vector"

class PopulationTable;

class Nn {
private:
	std::vector <float *> layers;
	std::vector <size_t> architecture;

public:
	Nn (const std::vector <size_t> & _architecture);

	friend PopulationTable;
};

#endif //NN_GPU_NN_HPP

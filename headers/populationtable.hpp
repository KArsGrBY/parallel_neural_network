#ifndef NN_GPU_POPULATIONTABLE_HPP
#define NN_GPU_POPULATIONTABLE_HPP

#include "nn.hpp"
#include "vector"

class PopulationTable {
private:
	std::vector <std::vector <float>> layers;
	size_t bestPersonIndex;
	const std::vector <size_t> * architecture;
	std::vector <Nn> population;

	PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population);
};

#endif //NN_GPU_POPULATIONTABLE_HPP

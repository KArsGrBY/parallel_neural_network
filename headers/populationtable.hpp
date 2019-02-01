#ifndef NN_GPU_POPULATIONTABLE_HPP
#define NN_GPU_POPULATIONTABLE_HPP

#include "ml.hpp"
#include "nn.hpp"
#include "vector"

namespace ml {
	struct PopulationTable {
		std::vector <std::vector <float>> weights;
		std::vector <std::vector <float>> bestWeights;
		std::vector <std::vector <float>> motions;
		size_t bestPersonIndex;
		const std::vector <size_t> * architecture;
		std::vector <Nn> population;

		PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population);
	};
}

#endif //NN_GPU_POPULATIONTABLE_HPP

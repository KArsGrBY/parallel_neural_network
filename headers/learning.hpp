#ifndef NN_GPU_LEARNING_HPP
#define NN_GPU_LEARNING_HPP

#include "ml.hpp"
#include "task.hpp"
#include "samplestable.hpp"
#include "populationtable.hpp"
#include "nn.hpp"
#include "vector"

namespace ml {

	class Learning {
	private:
		SamplesTable samplesTable;
		PopulationTable populationTable;
		std::vector <size_t> architecture;

	public:
		Learning (const std::vector <size_t> & _arcitecture, size_t _countOfNetworks,
				  const std::vector <std::pair <std::vector <float>, std::vector <float>>> & _samples);
	};
}


#endif //NN_GPU_LEARNING_HPP

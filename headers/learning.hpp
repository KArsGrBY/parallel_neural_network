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
		std::vector <size_t> architecture;
		PopulationTable populationTable;
		SamplesTable samplesTable;
		std::vector <Task> tasks;
		std::vector <float> bestErrors, errors;

	public:
		Learning (const std::vector <size_t> & _arcitecture, size_t _countOfNetworks,
				  const std::vector <std::pair <std::vector <float>, std::vector <float>>> & _samples);

		void iteration ();
	};
}


#endif //NN_GPU_LEARNING_HPP

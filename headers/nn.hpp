#ifndef NN_GPU_NN_HPP
#define NN_GPU_NN_HPP

#include "ml.hpp"
#include "vector"
#include "populationtable.hpp"

namespace ml {
	class Nn {
		friend PopulationTable;

	private:
		std::vector <std::vector <float>> weights;
		std::vector <size_t> architecture;

	public:
		Nn (const std::vector <size_t> & _architecture);
	};
}

#endif //NN_GPU_NN_HPP

#ifndef NN_GPU_NN_HPP
#define NN_GPU_NN_HPP

#include "ml.hpp"
#include "vector"
#include "populationtable.hpp"

namespace ml {
	class Nn {
		friend class PopulationTable;

	private:
		std::vector <std::vector <float> > weights;
		std::vector <std::vector <float> > bestWeights;
		std::vector <std::vector <float> > motions;
		std::vector <size_t> architecture;

	public:
		Nn (const std::vector <size_t> & _architecture);
	};
}

#endif //NN_GPU_NN_HPP

#ifndef NN_GPU_NN_HPP
#define NN_GPU_NN_HPP

#include "vector"

namespace ml {

	class PopulationTable;

	class Nn {
		friend PopulationTable;

	private:
		std::vector <float *> layers;
		std::vector <size_t> architecture;

	public:
		Nn (const std::vector <size_t> & _architecture);

	};
}

#endif //NN_GPU_NN_HPP

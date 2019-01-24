#ifndef NN_GPU_TASK_HPP
#define NN_GPU_TASK_HPP

#include "ml.hpp"
#include "CL/cl.hpp"
#include "vector"

namespace ml {


	class Task {
	private:
		cl::Device device;
		std::vector <cl::Buffer> neurons;
		std::vector <cl::Buffer> weights;
		std::vector <cl::Buffer> bestPerson;
		cl::Buffer outputs;
		size_t population;
		size_t firstIndex, lastIndex;

	public:
		Task (size_t _firstIndex, size_t _lastIndex);
	};
}

#endif //NN_GPU_TASK_HPP

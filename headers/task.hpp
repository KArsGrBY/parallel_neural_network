#ifndef NN_GPU_TASK_HPP
#define NN_GPU_TASK_HPP

#include "ml.hpp"
#include "CL/cl.hpp"
#include "vector"

namespace ml {
	class Task {
	private:
		static cl::Program::Sources srcExeLayer;
		static cl::Program::Sources srcActLayer;
		static cl::Program::Sources srcUpdate;

		cl::Device device;
		cl::Context context;
		cl::CommandQueue commandQueue;
		cl::Program progExeLayer;
		cl::Program progActLayer;
		cl::Program progUpdate;
		cl::Kernel kernelExeLayer;
		cl::Kernel kernelActLayer;
		cl::Kernel kernelUpdate;

		std::vector <cl::Buffer> neurons;
		std::vector <cl::Buffer> weights;
		std::vector <cl::Buffer> bestPerson;
		cl::Buffer outputs;
		size_t population;
		size_t firstIndex, lastIndex;

	public:
		Task (cl::Device _device, size_t firstIdx, size_t lstIdx, PopulationTable * popTable, SamplesTable * sampTamle);
	};
}

#endif //NN_GPU_TASK_HPP

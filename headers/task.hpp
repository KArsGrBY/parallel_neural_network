#ifndef NN_GPU_TASK_HPP
#define NN_GPU_TASK_HPP

#include "ml.hpp"
#include "CL/cl.hpp"
#include "vector"
#include "singlekernel.hpp"
#include "populationtable.hpp"
#include "samplestable.hpp"

namespace ml {
	class Task {
		friend class Learning;
	private:
		cl::Program::Sources srcExeLayer;
		cl::Program::Sources srcError;
		cl::Program::Sources srcUpdate;

		cl::Device device;
		cl::Context context;
		cl::CommandQueue commandQueue;
		cl::Program progExeLayer;
		cl::Program progError;
		cl::Program progUpdate;
		cl::Kernel kernelExeLayer;
		cl::Kernel kernelError;
		cl::Kernel kernelUpdate;

		std::vector <cl::Buffer> neurons;
		std::vector <cl::Buffer> weights;
		std::vector <cl::Buffer> bestPerson;

		const std::vector <size_t> *architecture;
		cl::Buffer outputs;
		size_t population, samples;
		size_t firstIndex, lastIndex;

	public:
		Task (cl::Device _device, size_t _firstIndex, size_t _lastIndex, PopulationTable * popTable, SamplesTable * sampTable);

		void executeLayer (size_t layer);
	};
}

#endif //NN_GPU_TASK_HPP

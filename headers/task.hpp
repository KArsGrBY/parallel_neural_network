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
		const cl::Program::Sources & srcExeLayer = SingletonKernel::getInstance().getExeLayerSource();
		const cl::Program::Sources & srcActLayer = SingletonKernel::getInstance().getActLayerSource();
		const cl::Program::Sources & srcUpdate = SingletonKernel::getInstance().getUpdateSource();

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

		const std::vector <size_t> *architecture;
		cl::Buffer outputs;
		size_t population, samples;
		size_t firstIndex, lastIndex;

	public:
		Task (const cl::Device & _device, size_t _firstIndex, size_t _lastIndex, PopulationTable * popTable, SamplesTable * sampTamle);

		void executeLayer (size_t layer);
	};
}

#endif //NN_GPU_TASK_HPP

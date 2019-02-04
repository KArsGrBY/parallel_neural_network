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
		cl::Program progFinalError;
		cl::Program progBestError;
		cl::Program progCopy;
		cl::Program progUpdate;

		cl::Kernel kernelExeLayer;
		cl::Kernel kernelError;
		cl::Kernel kernelFinalError;
		cl::Kernel kernelBestError;
		cl::Kernel kernelCopy;
		cl::Kernel kernelUpdate;

		std::vector <cl::Buffer> neurons;
		std::vector <cl::Buffer> weights;
		std::vector <cl::Buffer> bestWeights;
		std::vector <cl::Buffer> motions;
		std::vector <cl::Buffer> bestPerson;
		cl::Buffer errors;
		cl::Buffer finalErrors;
		cl::Buffer bestErrors;
		cl::Buffer outputs;

		std::vector <float> tempErrors;
		const std::vector <size_t> *architecture;
		size_t population, samples;
		size_t firstIndex, lastIndex;

	public:
		Task (cl::Device _device, size_t _firstIndex, size_t _lastIndex, PopulationTable * popTable, SamplesTable * sampTable, std::vector <float> & _errors);

		void executeLayer (size_t layer);

		void calculateError ();

		void calculateFinalError (std::vector <float> & error);

		void updatePersonsBestState (size_t layer);

		void downloadBestErrors (float * err);

		void downloadBestPerson (size_t layer, size_t personId, float * weights);

		void uploadBestPerson (size_t layer, float * cpuWeights);
	};
}

#endif //NN_GPU_TASK_HPP

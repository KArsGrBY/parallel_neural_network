#pragma comment( lib, "opencl.lib" )
#define __CL_ENABLE_EXCEPTIONS

#include "samplestable.hpp"
#include "populationtable.hpp"
#include "task.hpp"
#include "iostream"
#include "iomanip"

ml::Task::Task (cl::Device _device,
				size_t _firstIndex,
				size_t _lastIndex,
				ml::PopulationTable * popTable,
				ml::SamplesTable * sampTable,
				std::vector <float> & _errors) {
	architecture = popTable->architecture;

	firstIndex = _firstIndex, lastIndex = _lastIndex;
	population = lastIndex - firstIndex;
	samples = sampTable->size;
	device = _device;
	std::vector <cl::Device> devices = {device};

	context = cl::Context(devices);

	const auto & architecture = *popTable->architecture;
	neurons = std::vector <cl::Buffer>(architecture.size());

	//init kernel's source code
	srcExeLayer = cl::Program::Sources(1, std::make_pair(SingletonKernel::getInstance().getExeLayerCode().data(),
														 SingletonKernel::getInstance().getExeLayerCode().size() + 1));
	srcError = cl::Program::Sources(1, std::make_pair(SingletonKernel::getInstance().getErrorCode().data(),
													  SingletonKernel::getInstance().getErrorCode().size() + 1));
	srcUpdate = cl::Program::Sources(1, std::make_pair(SingletonKernel::getInstance().getUpdateCode().data(),
													   SingletonKernel::getInstance().getUpdateCode().size() + 1));

	//push output of samples into buffer
	outputs = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						 sampTable->output.size() * sizeof(float), sampTable->output.data());

	std::vector <float> buf(sampTable->input.size() * population);
	for (size_t index = 0; index < population; ++index) {
		std::copy(std::begin(sampTable->input), std::end(sampTable->input),
				  buf.data() + index * sampTable->input.size());
	}
	//push input of samples into buffer
	neurons[0] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float), buf.data());

	//push layers into buffer
	for (size_t layer = 1; layer < architecture.size(); ++layer) {
		buf = std::vector <float>(samples * architecture[layer] * population);
		neurons[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float),
									buf.data());
	}

	//set buffer for errors
	errors = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
						population * samples, _errors.data() + _firstIndex * samples);

	//push weights of nn's into buffer
	weights = std::vector <cl::Buffer>(architecture.size() - 1);
	bestWeights = std::vector <cl::Buffer>(architecture.size() - 1);
	motions = std::vector <cl::Buffer>(architecture.size() - 1);

	bestPerson = std::vector <cl::Buffer>(architecture.size() - 1);
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		size_t weightsNumber = architecture[layer] * architecture[layer + 1];

		weights[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
									weightsNumber * population * sizeof(float),
									popTable->weights[layer].data() + firstIndex * weightsNumber);

		bestWeights[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
										weightsNumber * population * sizeof(float),
										popTable->bestWeights[layer].data() + firstIndex * weightsNumber);

		motions[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
									weightsNumber * population * sizeof(float),
									popTable->motions[layer].data() + firstIndex * weightsNumber);

		//push weights of best nn into buffer
		bestPerson[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsNumber * sizeof(float),
									   popTable->weights[layer].data());
	}


	commandQueue = cl::CommandQueue(context, device);

	try {
		//init executive layer
		progExeLayer = cl::Program(context, srcExeLayer);
		progExeLayer.build(devices);
		kernelExeLayer = cl::Kernel(progExeLayer, "execute");


		//init activation layer
//		progError = cl::Program(context, srcError);
//		progError.build(devices);
//		kernelError = cl::Kernel(progError, "activate");
	} catch (cl::Error err) {
		std::cerr << err.what() << '\n' << progExeLayer.getBuildInfo <CL_PROGRAM_BUILD_LOG>(device);
	}

	/*
	//init update layer
	progUpdate = cl::Program(context, srcUpdate);
	progUpdate.build(devices);
	kernelUpdate = cl::Kernel(progUpdate, "update");
	 */
}

void ml::Task::executeLayer (size_t layer) {
	size_t sizeIn = (*architecture)[layer];
	size_t sizeOut = (*architecture)[layer + 1];

	size_t args = 0;
	kernelExeLayer.setArg(args++, neurons[layer]);
	kernelExeLayer.setArg(args++, neurons[layer + 1]);
	kernelExeLayer.setArg(args++, weights[layer]);
	kernelExeLayer.setArg(args++, (unsigned int) (*architecture)[layer]);
	kernelExeLayer.setArg(args++, (unsigned int) (*architecture)[layer + 1]);
	kernelExeLayer.setArg(args++, (unsigned int) samples);

	commandQueue.enqueueNDRangeKernel(kernelExeLayer, cl::NullRange, cl::NDRange(population, samples, sizeOut));
	commandQueue.finish();
}
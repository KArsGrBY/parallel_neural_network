#include "samplestable.hpp"
#include "populationtable.hpp"
#include "task.hpp"

ml::Task::Task (const cl::Device & _device, size_t _firstIndex, size_t _lastIndex, ml::PopulationTable * popTable,
				ml::SamplesTable * sampTamle) {
	architecture = popTable->architecture;

	firstIndex = _firstIndex, lastIndex = _lastIndex;
	population = lastIndex - firstIndex;
	samples = sampTamle->size;
	device = _device;
	std::vector <cl::Device> devices = {device};

	context = cl::Context(devices);

	//push output of samples into buffer
	outputs = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						 sampTamle->output.size() * sizeof(float), sampTamle->output.data());

	const auto & architecture = *popTable->architecture;

	neurons = std::vector <cl::Buffer>(architecture.size());


	std::vector <float> buf(sampTamle->input.size() * population);
	for (size_t index = 0; index < population; ++index) {
		std::copy(std::begin(sampTamle->input), std::end(sampTamle->input),
				  buf.data() + index * sampTamle->input.size());
	}
	//push input of samples into buffer
	neurons[0] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float), buf.data());

	//push layers into buffer
	for (size_t layer = 1; layer < architecture.size(); ++layer) {
		buf = std::vector <float>(sampTamle->size * architecture[layer] * population);
		neurons[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float),
									buf.data());
	}

	//push weights of nn's into buffer
	weights = std::vector <cl::Buffer>(architecture.size() - 1);
	bestPerson = std::vector <cl::Buffer>(architecture.size() - 1);
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		size_t weightsNumber = architecture[layer] * architecture[layer + 1];
		weights[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
									weightsNumber * population * sizeof(float),
									popTable->weights[layer].data() + firstIndex * weightsNumber);

		//push weights of best nn into buffer
		bestPerson[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsNumber * sizeof(float),
									   popTable->weights[layer].data());
	}


	commandQueue = cl::CommandQueue(context, device);


	//init executive layer
	progExeLayer = cl::Program(context, srcExeLayer);
	progExeLayer.build(devices);
	kernelExeLayer = cl::Kernel(progExeLayer, "execute");

	//init activation layer
	progActLayer = cl::Program(context, srcActLayer);
	progActLayer.build(devices);
	kernelActLayer = cl::Kernel(progActLayer, "activate");

	//init update layer
	progUpdate = cl::Program(context, srcUpdate);
	progUpdate.build(devices);
	kernelUpdate = cl::Kernel(progUpdate, "update");
}

#define inBlock		1
#define outBlock	1

void ml::Task::executeLayer (size_t layer) {
	size_t sizeIn = (*architecture)[layer];
	size_t sizeOut = (*architecture)[layer + 1];
	kernelExeLayer.setArg(0, sizeIn);
	kernelExeLayer.setArg(1, sizeOut);
	kernelExeLayer.setArg(2, weights[layer]);
	kernelExeLayer.setArg(3, (*architecture)[layer]);
	kernelExeLayer.setArg(4, (*architecture)[layer + 1]);
	kernelExeLayer.setArg(5, population);
	kernelExeLayer.setArg(6, inBlock);
	kernelExeLayer.setArg(7, outBlock);

	commandQueue.enqueueNDRangeKernel(kernelExeLayer, cl::NullRange, cl::NDRange(population, samples, sizeIn * sizeOut / inBlock / outBlock), 128);
}
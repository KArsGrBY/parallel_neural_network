#include <samplestable.hpp>
#include <populationtable.hpp>
#include "task.hpp"

ml::Task::Task (cl::Device _device, size_t firstIdx, size_t lstIdx, ml::PopulationTable * popTable,
				ml::SamplesTable * sampTamle) {
	firstIndex = firstIdx, lastIndex = lstIdx;
	population = lastIndex - firstIndex;
	device = _device;
	std::vector <cl::Device> devices = {device};

	context = cl::Context(devices);

	//push output of samples into buffer
	outputs = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						 sampTamle->sizeOut * sampTamle->size * sizeof(float), sampTamle->output.data());

	const auto & architecture = *popTable->architecture;

	neurons = std::vector <cl::Buffer>(architecture.size());


	std::vector <float> buf(sampTamle->sizeIn * sampTamle->size * population);
	for (size_t index = 0; index < population; ++index) {
		std::copy(std::begin(sampTamle->input), std::end(sampTamle->input), buf.data() + index * sampTamle->size);
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
		weights[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
									popTable->weights[layer].size() * sizeof(float), popTable->weights[layer].data());

		//push weights of best nn into buffer
		bestPerson[layer] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
								architecture[layer] * architecture[layer + 1] * sizeof(float),
								popTable->weights[layer].data());
	}


	commandQueue = cl::CommandQueue(context, device);


	/*
	//init executive layer
	progExeLayer = cl::Program(context, source);
	progExeLayer.build(devices);
	kernelExeLayer = cl::Kernel(progExeLayer, "execute");

	//init activation layer
	progActLayer = cl::Program(context, source);
	progActLayer.build(devices);
	kernelActLayer = cl::Kernel(progActLayer, "execute");

	//init update layer
	progUpdate = cl::Program(context, source);
	progUpdate.build(devices);
	kernelUpdate = cl::Kernel(progUpdate, "execute");
	*/
}
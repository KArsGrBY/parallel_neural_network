#include "learning.hpp"
#include "cassert"

ml::Learning::Learning (const std::vector <size_t> & _arcitecture, size_t _countOfNetworks,
						const std::vector <std::pair <std::vector <float>, std::vector <float>>> & _samples)
		: architecture(_arcitecture),
		  populationTable(PopulationTable(&architecture, std::vector <Nn>(_countOfNetworks, Nn(architecture)))),
		  samplesTable(SamplesTable(architecture.front(), architecture.back())),
		  tasks(std::vector <Task>()) {
	if (architecture.size() < 2) {
		assert("Number layers in neural network must be at least 2");
	}

//	for (int i = 0; i < populationTable.weights[0].size(); i++) {
//		std::cout << populationTable.weights[0][i] << '\n';
//	}
//	std::cout << std::endl << std::endl;

	for (const auto & sample : _samples) {
		samplesTable.addSample(sample.first, sample.second);
	}

	std::vector <cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector <cl::Device> devices;
	for (auto & platform : platforms) {
		std::vector <cl::Device> curDevices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &curDevices);
		std::copy(std::begin(curDevices), std::end(curDevices), std::back_inserter(devices));
	}

	size_t personIndex = 0;
	size_t personsInTask = _countOfNetworks / devices.size() + (_countOfNetworks % devices.size() != 0);

	for (const auto & device : devices) {
		size_t endOfGroup = personIndex + personsInTask;
		if (endOfGroup > _countOfNetworks) {
			endOfGroup = _countOfNetworks;
		}
		tasks.emplace_back(device, personIndex, endOfGroup, &populationTable, &samplesTable);
		personIndex = endOfGroup;
		std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		break; // for debug
	}
}

void ml::Learning::iteration () {
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		for (auto & task : tasks) {
			task.executeLayer(layer);
		}
		break; // for debug
	}
}
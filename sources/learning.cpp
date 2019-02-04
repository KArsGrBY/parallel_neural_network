#include "learning.hpp"
#include "algorithm"
#include "cassert"

std::vector <ml::Nn> generateNetworks (size_t number, const std::vector <size_t> & arcitecture) {
	std::vector <ml::Nn> networks;
	networks.reserve(number);
	for (size_t id = 0; id < number; ++id) {
		networks.push_back(ml::Nn(arcitecture));
	}
	return networks;
}

ml::Learning::Learning (const std::vector <size_t> & _arcitecture, size_t _countOfNetworks,
						const std::vector <std::pair <std::vector <float>, std::vector <float>>> & _samples)
		: architecture(_arcitecture),
		  populationTable(PopulationTable(&architecture, generateNetworks(_countOfNetworks, _arcitecture))),
		  samplesTable(SamplesTable(architecture.front(), architecture.back())),
		  tasks(std::vector <Task>()),
		  bestErrors(std::vector <float>()),
		  errors(std::vector <float>()) {

	if (architecture.size() < 2) {
		assert("Number layers in neural network must be at least 2");
	}

	//init neural networks
	for (size_t person = 0; person < _countOfNetworks; ++person) {

	}

	for (const auto & sample : _samples) {
		samplesTable.addSample(sample.first, sample.second);
	}

	errors = std::vector <float>();
	bestErrors = std::vector <float>();

	errors.reserve(_countOfNetworks);
	bestErrors.reserve(_countOfNetworks);

	for (size_t personId = 0; personId < _countOfNetworks; ++personId) {
		errors.push_back(populationTable.population[personId].curError);
		bestErrors.push_back(populationTable.population[personId].bestError);
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
		tasks.emplace_back(device, personIndex, endOfGroup, &populationTable, &samplesTable, errors);
		personIndex = endOfGroup;
		std::cout << device.getInfo <CL_DEVICE_NAME>() << std::endl;
		break; // for debug
	}
}

void ml::Learning::iteration () {
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		for (auto & task : tasks) {
			task.executeLayer(layer);
		}
	}

	for (auto & task : tasks) {
		task.calculateError();
	}

	errors.clear();
	for (auto & task : tasks) {
		task.calculateFinalError(errors);
	}

	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		for (auto & task : tasks) {
			task.updatePersonsBestState(layer);
		}
	}

	size_t counter = 0;
	for (auto & task : tasks) {
		task.downloadBestErrors(bestErrors.data() + counter);
		counter += task.population;
	}

	size_t taskWithBestPerson = 0;
	size_t bestPersonId = std::min_element(std::begin(bestErrors), std::end(bestErrors)) - std::begin(bestErrors);
	for (; tasks[taskWithBestPerson].population <= bestPersonId;
		   bestPersonId -= tasks[taskWithBestPerson++].population);


	std::vector <std::vector <float>> bestWeights((int) architecture.size() - 1);
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		bestWeights[layer] = std::vector <float> (architecture[layer] * architecture[layer + 1]);
		tasks[taskWithBestPerson].downloadBestPerson(layer, bestPersonId, bestWeights[layer].data());
	}

	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		for (auto & task : tasks) {
			task.uploadBestPerson(layer, bestWeights[layer].data());
		}
	}
}
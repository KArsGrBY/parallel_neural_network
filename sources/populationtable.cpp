#include "populationtable.hpp"

ml::PopulationTable::PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population) {
	architecture = _architecture;
	// number of layers must be greater than 1
	layers = std::vector <std::vector <float>>(architecture->size() - 1);
	bestPersonIndex = 0;
	population = _population;

	for (size_t layer = 0; layer + 1 < architecture->size(); ++layer) {
		size_t input = (*architecture)[layer], output = (*architecture)[layer + 1];
		auto & weights = layers[layer];
		weights.resize(input * output * population.size());
		for (size_t personIndex = 0; personIndex < population.size(); ++personIndex) {
			auto & person = population[personIndex].layers[layer];
			auto data = weights.data() + layer * input * output;
			std::copy(std::begin(person), std::end(person), data);
		}
	}
}

void ml::PopulationTable::addPersons (const std::vector <ml::Nn> & _population) {
	size_t oldPopulation = population.size();
	std::copy(std::begin(_population), std::end(_population), std::back_inserter(population));

	for (size_t layer = 0; layer + 1 < architecture->size(); ++layer) {
		size_t input = (*architecture)[layer], output = (*architecture)[layer + 1];
		auto & weights = layers[layer];
		weights.resize(input * output * population.size());
		for (size_t personIndex = oldPopulation; personIndex < population.size(); ++personIndex) {
			auto & person = population[personIndex].layers[layer];
			auto data = weights.data() + layer * input * output;
			std::copy(std::begin(person), std::end(person), data);
		}
	}
}
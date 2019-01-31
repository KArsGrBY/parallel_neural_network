#include "populationtable.hpp"
#include "iostream"

ml::PopulationTable::PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population) {
	architecture = _architecture;
	// number of layers must be greater than 1
	weights = std::vector <std::vector <float>>(architecture->size() - 1);
	bestPersonIndex = 0;
	population = _population;

	for (size_t layer = 0; layer + 1 < architecture->size(); ++layer) {
		size_t input = (*architecture)[layer], output = (*architecture)[layer + 1];
		auto & curWeights = weights[layer];
		curWeights.reserve(input * output * population.size());
		for (size_t personIndex = 0; personIndex < population.size(); ++personIndex) {
			auto & person = population[personIndex].weights[layer];
			std::copy(std::begin(person), std::end(person), std::back_inserter(curWeights));
		}
	}
}
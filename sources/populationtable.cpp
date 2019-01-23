#include "populationtable.hpp"

PopulationTable::PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population) {
	architecture = _architecture;
	// number of layers must be greater than 1
	layers = std::vector <std::vector <float>>(architecture->size() - 1);
	bestPersonIndex = 0;
	population = _population;

	for (size_t layer = 0; layer + 1 < architecture->size(); ++layer) {
		size_t input = (*architecture)[layer], output = (*architecture)[layer + 1];
		auto & weights = layers[layer];
		weights.resize(input * output * population.size());
		for (int personIndex = 0; personIndex < population.size(); ++personIndex) {
			auto & person = population[personIndex];
			auto data = weights.data() + layer * input * output;
			std::copy(person.layers[layer], person.layers[layer] + input * output, data);
			person.layers[layer] = data;
		}
	}
}
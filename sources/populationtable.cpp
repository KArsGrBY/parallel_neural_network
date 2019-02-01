#include "populationtable.hpp"
#include "iostream"

ml::PopulationTable::PopulationTable (std::vector <size_t> * _architecture, const std::vector <Nn> & _population) {
	architecture = _architecture;
	// number of layers must be greater than 1

	using vecs = std::vector <std::vector <float>>;
	weights = vecs(architecture->size() - 1);
	bestWeights = vecs(architecture->size() - 1);
	motions = vecs(architecture->size() - 1);

	bestPersonIndex = 0;
	population = _population;

	for (size_t layer = 0; layer + 1 < architecture->size(); ++layer) {
		size_t input = (*architecture)[layer], output = (*architecture)[layer + 1];

		auto & curWeights = weights[layer];
		auto & curBestWeights = bestWeights[layer];
		auto & curMotions = motions[layer];

		curWeights.reserve(input * output * population.size());
		curBestWeights.reserve(input * output * population.size());
		curMotions.reserve(input * output * population.size());
		for (size_t personIndex = 0; personIndex < population.size(); ++personIndex) {
			auto & _weights = population[personIndex].weights[layer];
			std::copy(std::begin(_weights), std::end(_weights), std::back_inserter(curWeights));

			auto & _bestWeights = population[personIndex].bestWeights[layer];
			std::copy(std::begin(_bestWeights), std::end(_bestWeights), std::back_inserter(curBestWeights));

			auto & _motions = population[personIndex].motions[layer];
			std::copy(std::begin(_motions), std::end(_motions), std::back_inserter(curMotions));
		}
	}
}
#include "learning.hpp"

ml::Learning::Learning (const std::vector <size_t> & _arcitecture, size_t _countOfNetworks,
						const std::vector <std::pair <std::vector <float>, std::vector <float>>> & _samples)
		: architecture(_arcitecture),
		  populationTable(PopulationTable(&architecture, std::vector <Nn>(_countOfNetworks, Nn(architecture)))),
		  samplesTable(SamplesTable(architecture.front(), architecture.back())) {


	for (const auto & sample : _samples) {
		samplesTable.addSample(sample.first, sample.second);
	}
}
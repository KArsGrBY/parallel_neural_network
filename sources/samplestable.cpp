#include "samplestable.hpp"

ml::SamplesTable::SamplesTable (size_t _sizeIn, size_t _sizeOut) {
	sizeIn = _sizeIn, sizeOut = _sizeOut;

	size = 0;

	input = std::vector <float>();
	output = std::vector <float>();
}

void ml::SamplesTable::addSample (const std::vector <float> & _input, const std::vector <float> & _output) {
	//_input.size() and _output.size() must be equal sizeIn and sizeOut respectabely
	++size;
	std::copy(std::begin(_input), std::end(_input), std::back_inserter(input));
	std::copy(std::begin(_output), std::end(_output), std::back_inserter(output));
}
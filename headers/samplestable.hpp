#ifndef NN_GPU_SAMPLESTABLE_HPP
#define NN_GPU_SAMPLESTABLE_HPP

#include "ml.hpp"
#include "vector"

namespace ml {
	class SamplesTable {
		friend class Task;
		friend class Learning;
	private:
		std::vector <float> input;
		std::vector <float> output;
		size_t sizeIn, sizeOut, size;

	public:
		SamplesTable (size_t _sizeIn, size_t _sizeOut);

		void addSample (const std::vector <float> & _input, const std::vector <float> & _output);
	};
}

#endif //NN_GPU_SAMPLESTABLE_HPP

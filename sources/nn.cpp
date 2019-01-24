#include "nn.hpp"
#include "random"

inline float randomFromRange (float x, float y) {
	static std::mt19937 gen(time(nullptr));
	std::uniform_real_distribution dis(x, y);
	return dis(gen);
}

ml::Nn::Nn (const std::vector <size_t> & _architecture) {
	architecture = _architecture;
	weights = std::vector <std::vector <float>>((int) architecture.size() - 1);
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		size_t input = architecture[layer], output = architecture[layer + 1];
		weights[layer] = std::vector <float>(input * output);
//		 		RANGE SETS HERE. 6.f for sigmoid function
		float range = 6.f / input;
		for (size_t neuronInput = 0; neuronInput < input; ++neuronInput) {
			for (size_t neuronOutput = 0; neuronOutput < output; ++neuronOutput) {
				weights[layer][neuronInput * output + neuronOutput] = randomFromRange(-range, range);
			}
		}
	}
}
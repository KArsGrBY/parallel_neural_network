#include "nn.hpp"
#include "random"

inline float randomFromRange (float x, float y) {
	static std::mt19937 gen(time(0));
	std::uniform_real_distribution dis(x, y);
	return dis(gen);
}

Nn::Nn (const std::vector <size_t> & _architecture) {
	architecture = _architecture;
	layers = std::vector <float *>((int) architecture.size() - 1);
	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		size_t input = architecture[layer], output = architecture[layer + 1];
		layers[layer] = new float[input * output];
//		 		RANGE SETS HERE. 6.f for sigmoid function
		float range = 6.f / input;
		for (size_t neuronInput = 0; neuronInput < input; ++neuronInput) {
			for (size_t neuronOutput = 0; neuronOutput < output; ++neuronOutput) {
				layers[layer][neuronInput * output + neuronOutput] = randomFromRange(-range, range);
			}
		}
	}

}
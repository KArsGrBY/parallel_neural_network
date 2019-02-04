#include "CL/cl.hpp"
#include "learning.hpp"
#include "bits/stdc++.h"

using namespace std;

class Timer {
private:
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration <double, std::ratio <1> >;

	std::chrono::time_point <clock_t> m_beg;

public:
	Timer () : m_beg(clock_t::now()) {
	}

	void reset () {
		m_beg = clock_t::now();
	}

	double elapsed () const {
		return std::chrono::duration_cast <second_t>(clock_t::now() - m_beg).count();
	}
};

const int SAMPLES = 100;
const int ITER = 20;
const int SIZE_IN = 1000, SIZE_OUT = 30;


inline float randomFromRange (float x, float y) {
	static std::mt19937 gen(time(nullptr));
	std::uniform_real_distribution dis(x, y);
	return dis(gen);
}

inline std::vector <float> vgen (size_t len) {
	std::vector <float> v(len);
	for (int i = 0; i < v.size(); i++) {
		v[i] = randomFromRange(0, 1);
	}
	return v;
}

int main (int argc, char ** argv) {
	using vec = std::vector <float>;
	using sample = std::pair <vec, vec>;
	std::vector <sample> samples;
	for (int samp = 0; samp < SAMPLES; samp++) {
		samples.push_back(std::make_pair(vgen(SIZE_IN), vgen(SIZE_OUT)));
	}
	ml::Learning learning({SIZE_IN, 30, SIZE_OUT}, 256, samples);

	/*ml::Learning learning({2, 4, 1}, 128, {
			{{1, 1}, {0}},
			{{1, 0}, {1}},
			{{0, 1}, {1}},
			{{0, 0}, {0}}
	});*/

	std::cerr << "START\n";
	Timer timer;
	for (int iter = 0; iter < ITER; iter++) {
		std::cerr << iter << std::endl;
		learning.iteration();
	}
	std::cout << std::fixed << std::setprecision(5) << timer.elapsed() / ITER << std::endl;
}
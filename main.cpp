#include "CL/cl.hpp"
#include "learning.hpp"
#include "bits/stdc++.h"
using namespace std;

int main (int argc, char ** argv) {
	ml::Learning learning({2, 2, 1}, 128, {
		{{0, 0}, {0}},
		{{1, 1}, {0}},
		{{1, 0}, {1}},
		{{0, 1}, {1}}});
}
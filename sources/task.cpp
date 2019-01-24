#include "task.hpp"

ml::Task::Task (cl::Device _device, size_t firstIdx, size_t lstIdx, ml::PopulationTable * popTable, ml::SamplesTable * sampTamle) {
	firstIndex = firstIdx, lastIndex = lstIdx;
	population = lastIndex - firstIndex;
	device = _device;
}
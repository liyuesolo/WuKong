#include <algorithm>
#include <strstream>
#include "../include/SerDe.h"

using namespace cs2d;

FileHeader Serializer::read_header(std::ifstream& stream) {
	FileHeader header;
	stream.seekg(0);
	// obtain configuration
	stream.read(reinterpret_cast<char*>(&header), sizeof(header));

	if (memcmp(header.magic_number, magic_number, 4))
		throw BadFileTypeException();
	if (header.version != version)
		throw BadFileVersionException();

	std::cout
		<< "Header read (v=" << header.version << "). Frame size is " << header.frame_size
		<< std::endl;
	return header;
}

void Serializer::record_state(FrameInfo& frame_info, const SimConfig& sim_config) {
	// this will partly move to constructor
	// if file is empty, write header
	stream.seekp(0, std::ios_base::end);
	int pos = stream.tellp();
	if (pos == 0) {
		write_header();
		pos = stream.tellp();
	} else {
		/*
		// read the header and check if the file is compatible
		FileHeader header = read_header(stream);
		if (header.dofs_size != dofs_size)
			std::cerr << "Output file has a different #dofs! (" << header.dofs_size/8 << "!=" << dofs_size/8 << ") Aborting write frame." << std::endl;
		else if (header.frame_size != frame_size)
			std::cerr << "Output file has a different #params! (" << header.params_size/8 << "!=" << params_size/8 << ") Aborting write frame." << std::endl;
		if (header.frame_size != frame_size)
			std::cerr << "Output file has a different frame size! (" << header.frame_size << "!=" << frame_size << ") Aborting write frame." << std::endl;
			*/

	}
	// check if the last frame is corrupted
	const auto [frame, frame_offset] = offset_to_frame(pos);
	print_position();
	if (frame < 0 || frame_offset != 0) {
		std::cerr << "Bad output file: end is not on a frame boundary (o=" << frame_offset << "). Aborting write frame." << std::endl;
		return;
	}
	std::cout << "Frame size is" << frame_size << std::endl;
	print_position();
	std::cout << "Writing frame " << frame << std::endl;

	assert(cellSim.params.size() * sizeof(double) == params_size);
	assert(cellSim.coords_state.size() * sizeof(double) == dofs_size);

	static_assert(sizeof(frame_info) <= frame_info_size_max);
	static_assert(sizeof(sim_config) <= frame_metadata_size - frame_info_size_max);
	// write the frame
	// set the frame start marker
	std::memcpy(frame_info.frame_start_marker, frame_start_marker.data(), 4);
	stream.write(reinterpret_cast<const char*>(&frame_info), sizeof(frame_info));
	stream.write(zero_buffer.data(), frame_info_size_max - sizeof(frame_info));
	assert(std::get<1>(offset_to_frame(stream.tellp())) == frame_info_size_max);
	stream.write(reinterpret_cast<const char*>(&sim_config), sizeof(sim_config));
	stream.write(zero_buffer.data(), frame_metadata_size - frame_info_size_max - sizeof(sim_config));
	assert(std::get<1>(offset_to_frame(stream.tellp())) == frame_metadata_size);
	std::cout << "Frame header written ";
	print_position();
	stream.write(reinterpret_cast<const char*>(cellSim.params.data()), params_size)
		.write(reinterpret_cast<const char*>(cellSim.coords_state.data()), dofs_size);
	std::cout << "Frame written ";
	print_position();
	const auto [f, fo] = offset_to_frame(stream.tellp());
	assert(f == frame+1);
	assert(fo == 0);
	stream.flush();
}

void Serializer::print_position() {
	const int abs = stream.tellp();
	const int after_header = (abs - header_metadata_size);
	const int frame = after_header / frame_size;
	const int in_frame = after_header % frame_size;
	std::cout << std::dec << "Abs = " << abs << "  Frame=" << frame << "  In frame="<< in_frame << std::endl;
}

void Serializer::write_header() {
	std::cout << "Writing header... ";
	FileHeader header;
	std::memcpy(header.magic_number, magic_number, 4);
	header.version = version;
	header.frame_size = frame_size;
	header.params_size = params_size;
	header.dofs_size = dofs_size;
	header.cell_segments = cellSim.cell_segments;
	header.n_boundary = cellSim.n_boundary;
	header.n_cells = cellSim.n_cells;
	std::cout << "n_cells: " << std::dec << cellSim.n_cells
	<< ",n_boundary: " << cellSim.n_boundary
	<< ",cell_segments: " << cellSim.cell_segments
	<< std::endl;

	stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
	stream.flush();
	std::cout << "Written " << std::dec << sizeof(header) << " << bytes" << std::endl;
}

Deserializer Deserializer::load_file(const std::string& path) {
	std::ifstream stream(path, std::ios::binary|std::ios::in);
	stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	stream.seekg(0);
	FileHeader header = read_header(stream);
	Deserializer de(header, stream);
	return de;
}

FileHeader Deserializer::read_header(std::ifstream& stream) {
	FileHeader header;
	stream.seekg(0);
	// obtain configuration
	stream.read(reinterpret_cast<char*>(&header), sizeof(header));

	if (memcmp(header.magic_number, magic_number, 4))
		throw BadFileTypeException();
	if (header.version != version)
		throw BadFileVersionException();

	std::cout
		<< "Header read (v=" << header.version << "). Frame size is " << header.frame_size
		<< std::endl;
	return header;
}

CellSim2D Deserializer::load_state(int timestep, int static_step, int identifier) {
	// find the frame in frame_list

	std::tuple<int, int, int> tup{timestep, static_step, identifier};
	auto frame_it = std::find(frame_list.begin(), frame_list.end(), tup);
	if (frame_it == frame_list.end())
		throw NoSuchFrameException();
	const int frame_idx = frame_it - frame_list.begin();	
	std::cout << "Frame found (idx=" << frame_idx << ")" << std::endl;

	const int frame_start = frame_to_offset(frame_idx);

	// read frame metadata
	FrameInfo frame_info;
	stream.seekg(frame_start);
	stream.read(reinterpret_cast<char*>(&frame_info), sizeof(frame_info));
	stream.seekg(frame_start + frame_info_size_max);
	SimConfig sim_config;
	stream.read(reinterpret_cast<char*>(&sim_config), sizeof(sim_config));

	// read frame body (params & coords)
	std::cout << std::dec << "Reading frame body" << std::endl;
	stream.seekg(frame_start + frame_metadata_size);
	const int params_count = params_size/8, dofs_count = dofs_size/8;
	std::vector<double> params(params_count), coords(dofs_count);
	stream.read(reinterpret_cast<char*>(params.data()), params_size);
	stream.read(reinterpret_cast<char*>(coords.data()), dofs_size);

	std::cout << "Instanciating CellSim2D" << std::endl;
	CellSim2D cs(
			sim_config, params, coords,
			header.cell_segments, header.n_boundary, header.n_cells);

	return cs;
}

FrameInfo Deserializer::read_frame_header(int i) {
	std::cout << "Read frame size is" << frame_size << std::endl;
	FrameInfo frame_info;
	// seek to frame
	const int frame_start = frame_to_offset(i);
	std::cout << "frame_start: " << std::hex << frame_start << std::endl;
	stream.seekg(frame_start);
	try {
		stream.read(reinterpret_cast<char*>(&frame_info), sizeof(FrameInfo));
	}
	catch (const std::ios_base::failure& e) {
		std::cout << "Exception during reading frame start marker" << e.what() << std::endl;
		if (stream.eof()) {
			std::cout << "end of file reached" << std::endl;
			stream.clear();
			throw NoFrameException();
		}
	}
	// check for the frame start marker
	//
	if(memcmp(frame_info.frame_start_marker, frame_start_marker.data(), 4))
		throw InvalidFrameException();
	return frame_info;
}

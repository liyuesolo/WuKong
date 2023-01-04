#ifndef CS2D_SERDE_H
#define CS2D_SERDE_H
/*
 * A frame consists of:
 * Offset	Size	Type		Content
 * 0		4		char[4]		Frame Marker (CSFS)
 * 4		252					Frame Metadata
 * 512		SP		double[]	Params
 * 512+SP	SD		double[]	Dofs
 *
 * The frame metadata contains:
 * Offset	Size	Content
 * 0		<=256	FrameInfo (results)
 * 256		<=256	SimConfig (from before the frame)
 *
 * The whole file consists of:
 * Offset	Size	Content
 * 0		4		Magic Number (CS2D)
 * 4		4		Version (0002)
 * 8		4		Frame size
 * 12		4		Params count
 * 16		4		Dofs count
 *
 * FS+256	4		Coordinates length (int32)
 * FS+264	Arb.	Coordinates data (8B*n_dof)
 */

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include<nlohmann/json.hpp>

#include "config.h"
#include "CellSim.h"

namespace cs2d {
	using std::pair, std::vector;
	class FileHeader {
		public:
			char magic_number[4];
			uint32_t version;
			uint32_t frame_size;
			uint32_t params_size;
			uint32_t dofs_size;
			uint32_t cell_segments;
			uint32_t n_boundary;
			uint32_t n_cells;
	};

class CellSim2D;
	class CellSimSerDe {
		protected:
			constexpr static int32_t header_metadata_size = sizeof(FileHeader);
			constexpr static int32_t frame_info_size_max = 256;
			constexpr static int32_t frame_metadata_size = 512;
			constexpr static char magic_number[4]{'C', 'S', '2', 'D'};
			constexpr static std::array<char, 4> frame_start_marker {'C','S','F','S'};
			constexpr static uint32_t version = 2;

			const uint32_t params_size;
			const uint32_t dofs_size;
			const uint32_t frame_size;
			CellSimSerDe(unsigned int params_count, unsigned int dofs_count):
				params_size{(uint32_t) params_count * 8},
				dofs_size{(uint32_t) dofs_count * 8},
				frame_size{frame_metadata_size + params_size + dofs_size} { }

			inline int frame_to_offset(int frame) const {
				return frame * frame_size + header_metadata_size;
			}

			inline std::pair<int, int> offset_to_frame(int pos) const {
				const int frame_start_offset = pos - header_metadata_size;
				if (frame_start_offset < 0)
					return {-1, -1};
				const int frame = frame_start_offset / frame_size, frame_offset = frame_start_offset % frame_size;
				return {frame, frame_offset};
			}

			friend class FileHeader;
	};

	struct FrameInfo {
		char frame_start_marker[4];
		int32_t timestep;
		int32_t static_step;
		int32_t identifier;
		int32_t line_search_steps;
		int32_t hess_reg_steps;
		int32_t dummy1, dummy2;

		double collision_time;
		double total_potential;
		double perimeter_potential;
		double area_potential;
		double collision_potential;
		double boundary_collision_potential;
		double boundary_shape_potential;
		double adhesion_potential;
		double residual;
	};
	class Serializer: public CellSimSerDe {
		public:
			Serializer(const std::string& path, const CellSim2D& cellSim):
				CellSimSerDe(cellSim.params.size(), cellSim.coords_state.size()),
				cellSim{cellSim},
				stream(path, std::ios::binary|std::ios::app),
				zero_buffer(frame_metadata_size, 0) {
				}
			void record_state(FrameInfo& frame_info, const SimConfig& sim_config);
			void print_position();
			void write_header();

			const CellSim2D& cellSim;
			std::fstream stream;
			const vector<char> zero_buffer;
		private:
			FileHeader read_header(std::ifstream& stream);
	};

	//class NoFrameException: public std::exception {
	class NoFrameException {
		public:
			const char * what () const {
				return "No frames left";
			}
	};

	class BadFileTypeException {
		public:
			const char * what () const {
				return "Bad file type.";
			}
	};

	class BadFileVersionException {
		public:
			const char * what () const {
				return "Bad file version";
			}
	};

	class InvalidFrameException {
		public:
			const char * what () const {
				return "Invalid frame";
			}
	};

	class NoSuchFrameException {
		public:
			const char * what () const {
				return "Frame does not exist";
			}
	};

	class Deserializer: public CellSimSerDe {
		public:
			static Deserializer load_file(const std::string& path);
		private:
			Deserializer(FileHeader header, std::ifstream& stream):
				CellSimSerDe(header.params_size/8, header.dofs_size/8),
				header{header},
				stream{std::move(stream)} {
					for (int i = 0; ; ++i) {
						try {
							FrameInfo frame_info = read_frame_header(i);
							frame_list.push_back({
									frame_info.timestep, frame_info.static_step, frame_info.identifier});
						}
						catch (const NoFrameException& e) {
							std::cout << "Exception: " << e.what() << std::endl;
							break;
						}
					}
					std::cout << frame_list.size() << " frames found: " << std::endl;
					for (auto [t, s, id]: frame_list)
						std::cout << "(t=" << t << ",s=" << s << ",id=" << id << "), ";
					std::cout << std::endl;
				}
		public:
			CellSim2D load_state(int timestep, int static_step, int identifier = 0);

		private:
			static FileHeader read_header(std::ifstream& stream);
			FrameInfo read_frame_header(int i);
			const FileHeader header;
			std::ifstream stream;
			vector<tuple<int,int,int>> frame_list;
	};
}
#endif

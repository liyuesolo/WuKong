#ifndef CS2D_SLICE_H
#define CS2D_SLICE_H
#include <cassert>
#include <vector>
#include <iostream>

template <typename T>
class VectorSlice {
	public:
		VectorSlice(std::vector<T>& cont, int begin, int end): container{cont}, idx_begin{begin}, idx_end{end}  {
			assert(begin <= end);
		}
		T* data() const {
			return container.data() + idx_begin;
		}

		T* begin() {
			return container.data() + idx_begin;
		}

		T* end() {
			return container.data() + idx_end;
		}

		int size() const {
			return idx_end - idx_begin;
		}

	private:
		std::vector<T>& container;
		const int idx_begin, idx_end;
};

template <typename T>
class ConstVectorSlice {
	public:
		ConstVectorSlice(const std::vector<T>& cont, int begin, int end): container{cont}, idx_begin{begin}, idx_end{end}  { }
		const T* data() const {
			return container.data() + idx_begin;
		}

		const T* begin() const {
			return container.data() + idx_begin;
		}

		const T* end() const {
			return container.data() + idx_end;
		}

		int size() const {
			return idx_end - idx_begin;
		}

	private:
		const std::vector<T>& container;
		const int idx_begin, idx_end;
};

template <typename T>
class Slice {
	public:
		Slice();
		Slice(int begin, int end): idx_begin{begin}, idx_end{end} { }

		int size() const {
			return idx_end - idx_begin;
		}

			ConstVectorSlice<T> operator()(const std::vector<T>& container) const {
				return ConstVectorSlice<T>(container, idx_begin, idx_end);
			}

			VectorSlice<T> operator()(std::vector<T>& container) const {
				return VectorSlice<T>(container, idx_begin, idx_end);
			}

		int idx_begin, idx_end;

};

//using VectorSlice<std::vector<double>, double>;
#endif

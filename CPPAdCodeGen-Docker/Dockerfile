FROM ubuntu:20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Lisbon \ 
    apt-get install -y cmake \
                       cmake-curses-gui \
                       libclang-8-dev \
                       llvm-8-dev \
                       clang-8 \
                       gcc \
                       g++ \
					   libgtest-dev \
                       wget \
                       doxygen \
                       graphviz \
                       git && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Lisbon \
    apt-get install -y texlive-extra-utils \
                       texlive-extra-utils \
                       texlive-fonts-extra \
                       texlive-latex-recommended && \
    apt-get autoclean && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV CPPADCG_DEPS="cppadcg_deps"
ENV CPPAD_INSTALL="cppadcg_deps/CppAD/install"
ENV EIGEN_INSTALL="cppadcg_deps/install_eigen/include/eigen3"

RUN mkdir /workdir
WORKDIR /workdir

RUN mkdir -p "${CPPADCG_DEPS}" \
            mkdir -p "${CPPAD_INSTALL}" \
            mkdir -p "${EIGEN_INSTALL}"

RUN wget https://github.com/coin-or/CppAD/archive/20200000.2.tar.gz

RUN curdir=$(pwd) ; \
	tar -xzf "20200000.2.tar.gz" && \
	cd CppAD-20200000.2 && \
	mkdir build && cd build && \
	cmake -Dcppad_prefix:PATH="${curdir}/${CPPAD_INSTALL}" \
                  -Dcppad_cxx_flags="-Wall -std=c++11 -Wshadow" \
                  .. && \
	make install

RUN curdir=$(pwd) ; \
	cd "${curdir}/${CPPADCG_DEPS}" && \
	wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2 && \
	tar -jxf eigen-3.3.8.tar.bz2 && \
	cd eigen-3.3.8 && \
	cp -r Eigen "${curdir}/${EIGEN_INSTALL}/"

RUN wget https://github.com/joaoleal/CppADCodeGen/archive/refs/tags/v2.4.3.tar.gz && \
	tar -xzf v2.4.3.tar.gz

WORKDIR /workdir/CppADCodeGen-2.4.3

RUN curdir=$(pwd) ; \
	mkdir build && cd build && \
	cmake --debug-find -DCMAKE_BUILD_TYPE=RELEASE \
                  -DUSE_VALGRIND=OFF \
                  -DENABLE_THREAD_POOL_TESTS=OFF \
                  -DCPPAD_HOME="/workdir/${CPPAD_INSTALL}/include" \
                  -DEIGEN3_INCLUDE_DIR="/workdir/${EIGEN_INSTALL}" \
                  ..

RUN cd build && make install

To Do 
- [ ] Write frame data into results file
- [ ] Set segment size etc. from intereface
- [ ] Split up interface nicely
- [ ] Screencaps working
- [ ] Initialize a file before reading/writing
- [ ] Conservative CCD
- [ ] Adaptive setting of collision weights (There is a bug where collisions can occur, maybe also a stability problem)
- [ ] Menu indicator for converged simulation
- [ ] Use ImPlot for nice plots
- [ ] load config from JSON, write config to json
- [ ] Try clang
- [ ] Color map
- [ ] Catch file errors (e.g. no such file)
- [ ] Jiggling
- [X] Better use of RNG (no correlation between different RNGs)
- [X] Better use of RNG (better init, 1 per attribute)
- [X] Adhesion potential
- [X] Logistic limiting of adhesion potential (per edge)
- [X] graph progress of static solving
- [X] Individual cell colors based on param
- [X] Implement differentiable boundary
- [X] Find a nice way to organize differentiable and non-differentiable dofs
- [X] better separation of coords and cells
- [X] Scanline algorithm for speedup of CCD and collision potential computation
- [X] Remove cubic potential implementation
- [X] Line Search
- [X] Collision Detection
- [X] Fix jiggling
- [X] Bounding box collision speedup
- [X] Hexagonal initialization
- [X] Render cell lines
- [X] Fix hessian
- [X] Boundary potential
- [X] Don't stop at half step

- New setup
   * [X] Implement perimeter potential
   * [X] Implement area potential
   * [X] Implement collision detection
   * [X] Implement collision potential
   * [X] Implement jacobi checks for each potential
   * [X] Implement hessian checks for each potential
   * [ ] Implement efficient CCD
   * [ ] Test perimeter potential
   * [ ] Test area potential
   * [ ] Test collision potential

[X] jupyter ready
[X] plot search direction again
[X] check if fixing the area to positive solves the problem (nope)
[X] one function for computing all the potentials
[X] separate config for simulation and visualization
[—] add obstacles (that can change over steps)
[ ] compute line search output over the whole length
[ ] create plots in jupyter
[ ] handle floating-point exceptions
[ ] decent file loading
[ ] implement serialization to file & deserialization
[X] load simulation & save simulation
[ ] create headless exectutable
[ ] correct triangle area (/2)
[ ] when plotting search direction, include the jacobian in the search direction
[ ] statistics: potentials for each step and sub-step, residuals for each step
[ ] statistics: all major parameters for all cells
[ ] playback mode in igl interface
[ ] profile the code
[ ] valgrind
[ ] Interface: color by different attributes
[ ] compute area statistics
[ ] clang-tidy or clang-format

- Report
   * [ ] Start Overleaf document, add illustrations
- Console output
   * [X] Print Intersection time, #steps, Potentials
   * [X] Remove pointless prints
   * [ ] Collision Hessian can be bad: check smaller FD steps
- Interface improvements
   * [X] Reinitialize Simulation
   * [X] Play/Pause
   * [ ] Set all relevant parameters from the simulation
   * [X] Adjust scale
- Simulation
   * [X] Fewer steps in line search
   * [X] ~Vertex-Vertex collisions~
   * [X] Segment-wise L2 for perimeter
   * [ ] Principled approach for choosing collision penalty weight (IPC Supp A)
- Code & Setup
   * [X] Serialize & deserialize CellSim state
   * [X] Serialization allows different #segments
   * [ ] Check if the settings match before saving
   * [ ] Record experiments
   * [ ] Create new repo
   * [ ] Split up code into source files
   * [ ] Use correct git identity
   * [X] Timing setup
   * [X] Rewrite entire code
- Videos & Images
   * [ ] Fix ScreenCap Problem
   * [ ] Ability to render movies
   * [ ] Display text (time etc), perhaps in an overlay
   * [ ] Plot potential along the line
   * [ ] Individual color for each cell
- Next major steps
   * [X] Introduce adhesion
   * [ ] Prohibit self-intersections
- [ ] 
- [ ] 

Next version of the thing:
- use an std::vector for the vertices
- cells have very little significance in the code, dof's matter

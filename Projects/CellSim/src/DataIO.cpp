#include "../include/DataIO.h"

void DataIO::loadDataFromTxt(const std::string& filename)
{
    std::vector<int> int_data;
    std::vector<long int> long_int_data;
    std::vector<T> float_data;
    // std::unordered_map<int, std::vector<TV>> frame_xyz_data;
    
    std::ifstream in(filename);
    std::string first_line;
    std::getline(in, first_line);
    int t, x, y, z, track_id;
    long int cell_id, parent_id;
    T node_score, edge_score;
    std::string comma;
    while (in >> t >> comma >> z >> comma >> y >> comma 
        >> x >> comma >> cell_id >> comma >> parent_id
        >> comma >> track_id >> comma >> node_score >>  comma >> edge_score)
    {
        int_data.push_back(t);
        int_data.push_back(x);
        int_data.push_back(y);
        int_data.push_back(z);
        // track_id_std_vec.push_back(track_id);
        long_int_data.push_back(cell_id);
        long_int_data.push_back(parent_id);
        float_data.push_back(node_score);
        float_data.push_back(edge_score);
    }
    in.close();
    
    write_binary<VectorXi>("/home/yueli/Downloads/drosophila_data/drosophila_side2_time_xyz.dat", 
        Eigen::Map<VectorXi>(int_data.data(), int_data.size()));
    write_binary<VectorXli>("/home/yueli/Downloads/drosophila_data/drosophila_side2_ids.dat", 
        Eigen::Map<VectorXli>(long_int_data.data(), long_int_data.size()));
    write_binary<VectorXT>("/home/yueli/Downloads/drosophila_data/drosophila_side2_scores.dat", 
        Eigen::Map<VectorXT>(float_data.data(), float_data.size()));
}

void DataIO::loadDataFromBinary(const std::string& int_data_file, 
        const std::string& long_int_data_file,
        const std::string& float_data_file)
{
    VectorXli cell_parent_ids;
    VectorXi time_xyz;
    VectorXT score_data;
    read_binary<VectorXli>(long_int_data_file, cell_parent_ids);
    read_binary<VectorXi>(int_data_file, time_xyz);
    read_binary<VectorXT>(float_data_file, score_data);

    int n_time_stamp = time_xyz.rows() / 4;
    positions.resize(n_time_stamp * 3);
    cell_ids.resize(n_time_stamp);
    parent_ids.resize(n_time_stamp);
    time_stamp.resize(n_time_stamp);
    edge_scores.resize(n_time_stamp);
    node_scores.resize(n_time_stamp);
    tbb::parallel_for(0, n_time_stamp, [&](int i){
        positions.segment<3>(i * 3) = time_xyz.segment<3>(i * 4 + 1);
        time_stamp[i] = time_xyz[i * 4];
        cell_ids[i] = cell_parent_ids[i * 2 + 0];
        parent_ids[i] = cell_parent_ids[i * 2 + 1];
        node_scores[i] = score_data[i * 2 + 0];
        edge_scores[i] = score_data[i * 2 + 1];
    });
}

void DataIO::trackCells()
{
    int n_data_entries = time_stamp.rows();
    struct FrameData
    {
        IV xyz;
        long int cell_id, parent_id;
        T edge_score;
        FrameData(const IV& _xyz, long int _cell_id, long int _parent_id, T _edge_score) : 
            xyz(_xyz), cell_id(_cell_id), parent_id(_parent_id), edge_score(_edge_score) {}
    };
    
    // sort all data by time stamp
    std::unordered_map<int, std::vector<FrameData>> frame_map;
    for (int i = 0; i < n_data_entries; i++)
    {
        int frame = time_stamp[i];
        if (frame_map.find(frame) == frame_map.end())
            frame_map[frame] = { FrameData(
                positions.segment<3>(i * 3),
                cell_ids[i], parent_ids[i],
                edge_scores[i]
            ) };
        else
            frame_map[frame].push_back(FrameData(
                    positions.segment<3>(i * 3),
                    cell_ids[i], parent_ids[i],
                    edge_scores[i]
                ));
        
    }
    
    std::ofstream out("cell_statistics.txt");
    for (int i = 0; i < frame_map.size(); i++)
    {
        // auto data = frame_map[i];
        out << i << " " << frame_map[i].size() << std::endl;
    }
    out.close();
    // std::exit(0);
    // std::cout << cell_ids[841186] << " " << time_stamp[841186] << std::endl;
    // std::getchar();

    // std::cout << frame_map.size() << std::endl;
    int n_frames = frame_map.size();
    n_frames = 100;
    // this maps long int cell indices to int cell indices
    std::unordered_map<long int, int> long_int_int_map;
    std::unordered_map<long int, long int> linage;
    int new_nuclei_cnt = 0;

    for (int frame = 0; frame < n_frames; frame++)
    {
        int n_nuleus = frame_map[frame].size();
        std::unordered_map<long int, int> cell_id_map_current;
        std::vector<long int> visited;
        for (int ni = 0; ni < n_nuleus; ni++)
        {
            FrameData frame_data = frame_map[frame][ni];    
            long int parent_id = frame_data.parent_id;
            long int cell_id = frame_data.cell_id;
            
            IV position = frame_data.xyz;
            
            // time 0
            if (parent_id == -1 && frame == 0)
            // if (parent_id == -1)
            {
                Nucleus nucleus;
                nucleus.idx = new_nuclei_cnt;
                long_int_int_map[cell_id] = new_nuclei_cnt;
                nucleus.positions = position;
                nucleus.parent_idx = -1;
                nucleus.start_frame = frame;
                nucleus.end_frame = frame;
                nucleus.score = frame_data.edge_score;
                new_nuclei_cnt++;
                nuclei.push_back(nucleus);
            }
            else
            {
                if (long_int_int_map.find(parent_id) == long_int_int_map.end() && parent_id == -1)
                {
                    // std::cout << "this nucleus does have a parent" << std::endl;
                    // std::cout << "parent_id: " << parent_id << " frame " << frame << std::endl;
                    // std::getchar();
                }
                else
                {
                    // find the same nucleus but in previous time step
                    int nuleus_id = long_int_int_map[parent_id];
                    bool better_nucleus = false;
                    if (std::find(visited.begin(), visited.end(), parent_id) != visited.end())
                    {
                        if (nuclei[nuleus_id].score < frame_data.edge_score)
                        {
                            int n_pos = nuclei[nuleus_id].positions.rows();
                            nuclei[nuleus_id].positions.tail<3>() = position;
                            cell_id_map_current[cell_id] = nuleus_id;
                            nuclei[nuleus_id].score = frame_data.edge_score;
                        }
                    }
                    else
                    {
                        nuclei[nuleus_id].end_frame++;
                        int n_pos = nuclei[nuleus_id].positions.rows();
                        nuclei[nuleus_id].positions.conservativeResize(n_pos + 3);
                        nuclei[nuleus_id].positions.segment<3>(n_pos) = position;
                        cell_id_map_current[cell_id] = nuleus_id;
                        nuclei[nuleus_id].score = frame_data.edge_score;
                        visited.push_back(parent_id);
                    }
                    
                }   
            }
        }
        if (frame > 0)
            long_int_int_map = cell_id_map_current;
    }
    // std::cout << nuclei.size() << std::endl;
    // std::getchar();    
    int valid_cnt = 0;
    for (Nucleus nucleus : nuclei)
    {
        // std::cout << nucleus.idx << " " << nucleus.parent_idx << " "
        //     << nucleus.start_frame << " "
        //     << nucleus.end_frame << " " << nucleus.positions.rows() 
        //     << std::endl;
        if (nucleus.positions.rows() == n_frames * 3)
            valid_cnt++;
    }
    std::cout << valid_cnt << std::endl;   
    MatrixXi cell_trajectories(n_frames * 3, valid_cnt);
    valid_cnt = 0;
    for (Nucleus nucleus : nuclei)
    {
        if (nucleus.positions.rows() == n_frames * 3)
        {
            cell_trajectories.col(valid_cnt) = nucleus.positions;
            valid_cnt++;
        }
    }
    // write_binary<MatrixXi>("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat", 
    //     cell_trajectories);
}

void DataIO::loadTrajectories(const std::string& filename, MatrixXT& trajectories)
{
    MatrixXi cell_trajectories;
    read_binary<MatrixXi>(filename, cell_trajectories);

    int n_frames = cell_trajectories.rows() / 3;
    int n_nucleus = cell_trajectories.cols();

    // for (int frame = 0; frame < n_frames; frame++)
    // {
    //     std::ofstream out("frame" + std::to_string(frame) + ".obj");
    //     for (int i = 0; i < n_nucleus; i++)
    //         out << "v " << cell_trajectories.col(i).segment<3>(frame * 3).transpose() << std::endl;
    //     out.close();
    // }

    trajectories.resize(n_nucleus * 3, n_frames);
    tbb::parallel_for(0, n_frames, [&](int frame){
        for (int i = 0; i < n_nucleus; i++)
        {
            IV tmp = cell_trajectories.col(i).segment<3>(frame * 3);
            for (int d = 0; d < 3; d++)
                trajectories(i* 3 + d, frame) = T(tmp[d]);
        }
    });
    
}
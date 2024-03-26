#include "../include/FEMSolver.h"

template <int dim>
void FEMSolver<dim>::dragMiddle()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<dim>(i * dim);
        if ((x[0] > center[0] - 0.1 && x[0] < center[0] + 0.1) 
            && (x[2] < min_corner[2] + 1e-6))
        {
            dirichlet_data[i * dim + 2] = -1;
            f[i * dim + 2] = -100.0;
        }
    }
}


template<int dim>
void FEMSolver<dim>::addDirichletBC()
{
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     TV x = deformed.segment<dim>(i * dim);
    //     if(i>=0 && i<4)
    //     {
    //         dirichlet_data[i * dim + 0] = 0;
    //         dirichlet_data[i * dim + 1] = 0;
    //     }
    // }
    // return;
    // for(int i=0; i<master_nodes.size(); ++i)
    // {
    //     dirichlet_data[master_nodes[i] * dim] = 0;
    //     dirichlet_data[master_nodes[i] * dim + 1] = 0;
    // }
    if(SLIDING_TEST)
    {
        for(int i=0; i<num_nodes-1; ++i)
        {
            dirichlet_data[i * dim + 0] = 0;
            dirichlet_data[i * dim + 1] = 0;
            std::cout << "add dirichlet boundary to node " << i << std::endl;
        }
        return;
    }

    if(PULL_TEST)
    {
        for(int i=0; i<num_nodes; ++i)
        {
            TV x = undeformed.segment<dim>(i * dim);
            //if(fabs(x[0]+1.62706) < 1e-5  || fabs(x[1]+1.73404) < 1e-5 || fabs(x[1]-1.73404)<1e-5)
            //if(fabs(x[0]+14.9999) < 1e-5)
            if(fabs(x[0]+14.9999) < 1e-5)
            {
                dirichlet_data[i * dim + 0] = 0;
                dirichlet_data[i * dim + 1] = 0;
                //std::cout << "add dirichlet boundary to node " << i << std::endl;
            }
            // if(fabs(x[0]-2.62534-DISPLAYSMENT) < 1e-5)
            // {
            //     dirichlet_data[i * dim + 1] = 0;
            //     //std::cout << "add dirichlet boundary to node " << i << std::endl;
            // }
            // if(i<693)
            // {
            //         dirichlet_data[i * dim + 0] = 0;
            //         dirichlet_data[i * dim + 1] = 0;
            // }
            // if(i==693)
            // {
            //         dirichlet_data[i * dim + 1] = 0;
            // }
            // if(i == 3701 || i == 2400)
            // {
            //     dirichlet_data[i * dim + 1] = 0;
            // }
            // if(i == 306 || i == 305)
            // {
            //     dirichlet_data[i * dim + 1] = 0;
            // }
            // if(i == 1022 || i == 1021)
            // {
            //     dirichlet_data[i * dim + 1] = 0;
            // }
           
        }
        return;
    }


    for (int i = 0; i < num_nodes; i++)
    {
        //std::cout<<center.transpose()<<std::endl;
        center[0] = 0.0;
        TV x = undeformed.segment<dim>(i * dim);
        //if (i == 0|| i == 11)

        if(TEST)
        {
            if (fabs(x[1]-0.) < 1e-5)
            //if(true)
            {
                dirichlet_data[i * dim + 1] = 0;
                std::cout << "add dirichlet boundary to node " << i << std::endl;
            }
        }
        else
        {
            //if(use_virtual_spring)
            // {
            //     if (fabs(x[0]-0.) < 1e-5)
            //     //if(true)
            //     {
            //         //dirichlet_data[i * dim + 0] = 0;
            //         dirichlet_data[i * dim] = 0;
            //         //std::cout << "add dirichlet boundary to node " << i << std::endl;
            //     }
            // }
            //dirichlet_data[0] = 0;
            if (fabs(x[1]-0.) < 1e-5)
            //if(true)
            {
                //dirichlet_data[i * dim + 0] = 0;
                dirichlet_data[i * dim + 1] = 0;
                if (fabs(x[0]-2.) < 1e-5)
                    dirichlet_data[i * dim] = 0;
                std::cout << "add dirichlet boundary to node " << i << std::endl;
            }

            if (fabs(x[1]-2.) < 1e-5)
            {
                //dirichlet_data[i * dim + 0] = 0;
                dirichlet_data[i * dim + 1] = 0;
                dirichlet_data[i * dim] = 0;
                std::cout << "add dirichlet boundary to node " << i << std::endl;
            }

            // if (fabs(x[1]-2.) < 1e-5)
            // //if(true)
            // {
            //     //dirichlet_data[i * dim + 0] = 0;
            //     dirichlet_data[i * dim + 1] = 0;
            //     dirichlet_data[i * dim] = 0;
            //     std::cout << "add dirichlet boundary to node " << i << std::endl;
            // }

            // if (fabs(x[0]-0.) < 1e-5)
            // //if(true)
            // {
            //     //dirichlet_data[i * dim + 0] = 0;
            //     dirichlet_data[i * dim] = 0;
            //     std::cout << "add dirichlet boundary to node " << i << std::endl;
            // }

            // if (fabs(x[1]-4.) < 1e-5)
            // //if(true)
            // {
            //     //dirichlet_data[i * dim + 0] = 0;
            //     dirichlet_data[i * dim] = 0;
            //     std::cout << "add dirichlet boundary to node " << i << std::endl;
            // }
        }

        // if (fabs(x[0]-4.2442) < 1e-5 && x[1] > 1)
        // //if(true)
        // {
        //     dirichlet_data[i * dim + 0] = 0;
        //     std::cout << "add dirichlet boundary to node " << i << std::endl;
        // }

        // if (fabs(x[0]-0) < 1e-5)
        // //if(true)
        // {
        //     dirichlet_data[i * dim + 0] = 0;
        //     std::cout << "add dirichlet boundary to node " << i << std::endl;
        // }
    }
}

template<int dim>
void FEMSolver<dim>::addRotationalDirichletBC(double theta)
{
    penalty_pairs.clear();
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<dim>(i * dim);
        double a = x[0];
        double b = x[1];
        double c = x[2];

       //std::cout<<i<<" "<<deformed.segment<dim>(i * dim).transpose()<<std::endl;
        // double r = sqrt(pow(b,2)+pow(c,2));
        // double theta2 = atan2(c,b);
        // double ncy = r*cos((a+0.5)*theta+theta2);
        // double ncx = r*sin((a+0.5)*theta+theta2);
        // u(dim*i+1) = ncy-b; u(dim*i+2) = ncx-c; u(dim*i+0) = 0;
        //std::cout<<i<<" "<<deformed.segment<dim>(i * dim).transpose()<<std::endl;

        //if (fabs(a-4.06897) < 1e-3)
        //if (fabs(a+0.7) < 1e-3)
        //if (fabs(a-0) < 1e-3)
        //if (fabs(b+1.49898) < 1e-3)
        //if (i<8046)
        if(i>=4096)
        {
            // dirichlet_data[i * dim +0] = 0;
            // dirichlet_data[i * dim + 2] = 0;
            // dirichlet_data[i * dim + 1] = -theta;
            // if(b > 0)
            // {
            //     double r = sqrt(pow(b-0.2,2)+pow(c,2));
            //     double theta2 = atan2(c,b-0.2);
            //     double ncy = 0.2*cos(theta)+r*cos(theta+theta2);
            //     double ncx = 0.2*sin(theta)+r*sin(theta+theta2);
            //     dirichlet_data[i * dim + 1] = ncy-b;
            //     dirichlet_data[i * dim + 2] = ncx-c;
            //     dirichlet_data[i * dim + 0] = 0;
            //     //deformed(dim*i+1) = ncy; deformed(dim*i+2) = ncx; deformed(dim*i+0) = a;
            // }
            // else
            // {
            //     double r = sqrt(pow(b+0.2,2)+pow(c,2));
            //     double theta2 = atan2(c,b+0.2);
            //     double ncy = -0.2*cos(theta)+r*cos(theta+theta2);
            //     double ncx = -0.2*sin(theta)+r*sin(theta+theta2);
            //     dirichlet_data[i * dim + 1] = ncy-b;
            //     dirichlet_data[i * dim + 2] = ncx-c;
            //     dirichlet_data[i * dim + 0] = 0;
            //     //deformed(dim*i+1) = ncy; deformed(dim*i+2) = ncx; deformed(dim*i+0) = a;
            // }


            // double r = sqrt(pow(b,2)+pow(c,2));
            // double theta2 = atan2(c,b);
            // double ncy = r*cos(theta+theta2);
            // double ncx = r*sin(theta+theta2);
            // dirichlet_data[i * dim + 1] = ncy-b;
            // dirichlet_data[i * dim + 2] = ncx-c;
            // dirichlet_data[i * dim + 0] = 0;

            // double r = sqrt(pow(b-0.5,2)+pow(c-0.01,2));
            // double theta2 = atan2(c-0.01,b-0.5);
            // double ncy = r*cos(theta+theta2)+0.5;
            // double ncx = r*sin(theta+theta2)+0.01;
            
            //deformed(dim*i+1) = ncy; deformed(dim*i+2) = ncx; deformed(dim*i+0) = a;
            // dirichlet_data[i * dim + 1] = ncy-b;
            // dirichlet_data[i * dim + 2] = ncx-c;
            // dirichlet_data[i * dim + 0] = 0;

            //deformed(dim*i+1) = ncy; deformed(dim*i+2) = ncx; deformed(dim*i+0) = a;

            // double r = sqrt(b*b+c*c);
            // double theta1 = atan2(c,b);
            // double nb = r*cos(theta+theta1);
            // double nc = r*sin(theta+theta1);
            // //deformed(dim*i+1) = nb; deformed(dim*i+2) = nc; deformed(dim*i+0) = a;
            // dirichlet_data[i * dim +1] = nb-b;
            // dirichlet_data[i * dim + 2] = nc-c;
            // dirichlet_data[i * dim + 0] = 0;
            //undeformed(dim*i+0) = nb; undeformed(dim*i+1) = nc; undeformed(dim*i+2) = a;
            //penalty_pairs.push_back(std::pair<int,T>(i * dim + 0, a-theta));
            //penalty_pairs.push_back(std::pair<int,T>(i * dim + 1, b));
            //penalty_pairs.push_back(std::pair<int,T>(i * dim + 2, c));
            f[i*dim+2] = 0.1*FORCE;
        }

        //if(fabs(x[0]+3.84528)<1e-3)
       // if(fabs(x[0]-1.0)<1e-3)
        //if(i<2747)
        //if(b<-2.63)
        //if(fabs(a-0.65)<1e-3)
        //if(fabs(a-0.65)<1e-3)
        if(i<4096)
        {
            dirichlet_data[i * dim + 0] = 0;
            dirichlet_data[i * dim + 1] = 0;
            dirichlet_data[i * dim + 2] = 0;

            // double r = sqrt(pow(b-0.5,2)+pow(c-0.01,2));
            // double theta2 = atan2(c-0.01,b-0.5);
            // double ncy = r*cos(-theta+theta2)+0.5;
            // double ncx = r*sin(-theta+theta2)+0.01;

            // //deformed(dim*i+1) = ncy; deformed(dim*i+2) = ncx; deformed(dim*i+0) = a;

            // penalty_pairs.push_back(std::pair<int,T>(i * dim + 0, a+theta));
            // penalty_pairs.push_back(std::pair<int,T>(i * dim + 1, b));
            // penalty_pairs.push_back(std::pair<int,T>(i * dim + 2, c));

            //f[i*dim+0] = 2*FORCE;
        }

        
    }
}

template<int dim>
void FEMSolver<dim>::addDirichletBCFromSTL()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<dim>(i * dim);
        if (dim == 3 && USE_SHELL)
        {
            if(i<4225 && (fabs(x[2]+0.5) < 1e-3||fabs(x[2]-0.5) < 1e-3|| fabs(x[0]+0.5) < 1e-3||fabs(x[0]-0.5) < 1e-3))
            //if(i<289)
            //if( i<169 && fabs(x[1]-0.47) < 0.01)
            //if(i<4225 && fabs(x[0]*x[0]+x[2]*x[2]-0.64) < 1e-3)
            //if(i<4225)
            {
                dirichlet_data[i * dim +0] = 0;
                dirichlet_data[i * dim + 1] = 0;
                dirichlet_data[i * dim + 2] = 0;
                //std::cout<<i<<std::endl;
            }
            // if(i == 366-289+289||i==367-289+289||i==362-289+289)
            // {
            //     dirichlet_data[i * dim +0] = 0;
            //     //dirichlet_data[i * dim + 1] = 0;
            //     dirichlet_data[i * dim + 2] = 0;
            // }
            continue;
        }
        

        //dirichlet_data[0] = 0;

        if(TEST_CASE == 3)
        {
            if (fabs(x[1]+0.9) < 1e-5 || fabs(x[1]-2.9) < 1e-5)
            {
                dirichlet_data[i * dim +0] = 0;
                dirichlet_data[i * dim + 1] = 0;
            }
            if(i == 254 || i == 252)
            {
                dirichlet_data[i * dim + 1] = 0;
            }
        }
        else
        {
            if (dim == 2 && fabs(x[1]+1.0) < 1e-5)
            {
                //dirichlet_data[i * dim + 0] = 0;
                dirichlet_data[i * dim + 1] = 0;
                if(TEST_CASE == 0)
                {
                    if(fabs(x[0]-0.0852436) < 1e-5)
                        dirichlet_data[i * dim] = 0;
                }
            }
            if(TEST_CASE == 0)
            {
                if(dim == 2 && (i == 154 || i == 149))
                {
                    //std::cout<<"add dirichlet BC"<<std::endl;
                    dirichlet_data[i * dim] = 0;
                }

                if(dim == 3 && fabs(x[1]) < 1e-5)
                    dirichlet_data[i * dim + 1] = 0;

                if(dim == 3 && (i == 30 || i == 108 || i == 0 || i == 1 || i == 78 || i == 79))
                {
                    dirichlet_data[i * dim] = 0;
                    dirichlet_data[i * dim+2] = 0;
                }
            }
            if(TEST_CASE == 1)
            {
                //if(i == 1 || i == 62 || i == 604 || i == 605 || i == 603)
                if(dim == 2)
                {
                    if(i == 0 || i == 2 || i == 690 || i == 691 || i == 689)
                    {
                        std::cout<<"add dirichlet BC"<<std::endl;
                        dirichlet_data[i * dim] = 0;
                    }
                }
                else if(dim == 3)
                {
                    if(fabs(x[1]) < 1e-5)
                        dirichlet_data[i * dim + 1] = 0;
                    // for(int j=0; j<master_nodes_3d.size(); ++j)
                    // {
                    //     if(master_nodes_3d[j].find(i) != master_nodes_3d[j].end())
                    //     {
                    //         dirichlet_data[i * dim + 0] = 0;
                    //         dirichlet_data[i * dim + 1] = 0;
                    //         dirichlet_data[i * dim + 2] = 0;
                    //     }
                    // }
                        
                    //if(i == 1 || i == 94 || i == 3 || i == 4 || i == 97 || i == 119)
                    if(i == 1 || i == 296 || i == 2 || i == 0 || i == 298 || i == 299)
                    {
                        dirichlet_data[i * dim] = 0;
                        dirichlet_data[i * dim+2] = 0;
                    }
                }
                
                
            }
            else if(TEST_CASE == 2)
            {
                if(i == 0 || i == 62 || i == 604 || i == 605 || i == 603)
                    dirichlet_data[i * dim] = 0;
            }
        }
        
       
        // if(fabs(x[0]-DISPLAYSMENT) < 1e-5)
        //     dirichlet_data[i * dim] = 0;
    }
}

template<int dim>
void FEMSolver<dim>::addNeumannBC()
{
    // f.setZero();
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     TV x = deformed.segment<dim>(i * dim);
    //     if(i==16 || i==19)
    //     {
    //         f[i * dim] += FORCE;
    //     }
    // }
    // return;

    f.setZero();
    int num_x_2 = WIDTH_2*SCALAR*RES_2+1, num_y_2 = HEIGHT_2*SCALAR*RES_2+1;
    std::vector<int> contact_vertices;
    int slave_index = 0;
    int master_index = 0;
    int master_out = 0;
    if(!PULL_TEST) master_out = boundary_info.back().master_index_1;
    int case_now = 0;
    //std::cout<<slave_nodes.size()<<" "<<master_nodes.size()<<" "<<boundary_info.size()<<std::endl;
    //std::cout<<deformed.size()<<" "<<std::endl;

    int index_1 = 0, index_2 = 0;
    spring_indices.clear();
    spring_ends.clear();

    if(PULL_TEST)
    {
        for (int i = 0; i < num_nodes; i++)
        {
            TV x = deformed.segment<dim>(i * dim);
            //if (fabs(x[0]-2.62534-DISPLAYSMENT) < 1e-5)
            if(i == 3704||i == 3698)
            //if(i == 323||i == 322)
            //if(i == 1070 || i == 1067)
            {
                f[i * dim] += spring_length*virtual_spring_stiffness;
                // std::cout<<FORCE<<std::endl;
                // spring_indices.push_back(i);
                // spring_ends.push_back(Eigen::Vector2d(x[0]+spring_length, x[1]));
            }
        }
        return;
    }

    if(!TEST)
    {
        // Find the segments indices

        // for(int i=0; i<boundary_info.size(); ++i)
        // {
        //     std::cout<<boundary_info[i].slave_index<<std::endl;
        // }
        // std::cout<<std::endl;
        int num_x = WIDTH_1*RES+1, num_y = HEIGHT_1*RES+1;
        int num_x_2 = WIDTH_2*SCALAR*RES_2+1, num_y_2 = HEIGHT_2*SCALAR*RES_2+1;

        for(int i=0; i<num_nodes; ++i)
        {
            TV x = undeformed.segment<dim>(i*dim);
            if(i == num_x*num_y+num_x_2-1)
            {
                //std::cout<<num_x*num_y+num_x_2-1<<std::endl;
                spring_indices.push_back(i);
                spring_ends.push_back(Eigen::Vector2d(x[0]+spring_length, x[1]));
            }
        }
        

        while(master_index<master_nodes.size())
        {
            double slave_x = deformed(2*slave_nodes[slave_index]);
            double master_x = deformed(2*master_nodes[master_index]);
            if(case_now == 0)
            {
                if(master_x <= slave_x)
                {
                    //contact_vertices.push_back(master_nodes[master_index]);
                    master_index ++;
                }
                else
                {
                    index_1 = master_nodes[master_index-1];
                    case_now = 1;
                    while(master_nodes[master_index] != master_out) master_index++;
                    index_2 = master_nodes[master_index];

                    //std::cout<<master_out<<" "<<master_nodes[master_index]<<std::endl;
                    //std::cout<<(deformed(2*boundary_info.back().master_index_2) == deformed(2*slave_nodes[slave_nodes.size()-1]))<<" "<<boundary_info.back().master_index_2<<std::endl;
                    //if(MODE==2) contact_vertices.push_back(boundary_info.back().master_index_2);
                }
            }
            else if(case_now == 1)
            {
                if(slave_index<slave_nodes.size()-1)
                {
                    contact_vertices.push_back(slave_nodes[slave_index]+(num_y_2-1)*num_x_2);
                    slave_index ++;
                }
                else
                {
                    contact_vertices.push_back(slave_nodes[slave_index]+(num_y_2-1)*num_x_2);
                    case_now = 2;
                }
            }
            else if(case_now == 2)
            {
                //contact_vertices.push_back(master_nodes[master_index]);
                master_index ++;
            }
                
        }
        std::cout<<"Contact vertices: ";
        for(int i=0; i<contact_vertices.size(); ++i)
        {
            std::cout<<contact_vertices[i]<<" ";
        }
        std::cout<<std::endl;

    }
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = deformed.segment<dim>(i * dim);
        center[0] = 0.0;
        //if (!(x[0] > center[0] - 0.1 && x[0] < center[0] + 0.1) 
        //    && (x[2] < min_corner[2] + 1e-6))
        //if(i == 10 || i == 21)

        
        if (TEST)
        {
            std::vector<std::pair<int,int>> segments = {std::pair<int,int>(15,16),std::pair<int,int>(16,17),std::pair<int,int>(17,18),std::pair<int,int>(18,19)};
            // if (fabs(x[1]- HEIGHT_1 - HEIGHT_2 - GAP) < 1e-7)
            // //if(i >= 1100 && i<1110)
            // {
            //     //for (int d = 0; d < dim; d++)
            //     if (fabs(x[0]- DISPLAYSMENT) < 1e-7 || fabs(x[0]- DISPLAYSMENT - WIDTH_2) < 1e-7)
            //     {
            //         f[i * dim + 1] = -FORCE/RES;
            //     }
            //     else
            //         f[i * dim + 1] = -2*FORCE/RES;

            //     std::cout << "add force to node " << i <<" with "<<f[i * dim + 1]<< std::endl;
            // }
            for(int i=0; i<segments.size(); ++i)
            {
                int i1 = segments[i].first;
                int i2 = segments[i].second;

                double len = (deformed.segment<2>(2*i2)-deformed.segment<2>(2*i1)).norm()/2;
                f[i1 * dim + 1] += -FORCE*len;
                f[i2 * dim + 1] += -FORCE*len;
            }
        }
        else
        {
            // if (fabs(x[1]-WIDTH_1-WIDTH_2-1e-4) < 1e-7)
            // //if(i >= 1100 && i<1110)
            // {
            //     //for (int d = 0; d < dim; d++)
            //     if (fabs(x[0]-DISPLAYSMENT) < 1e-7 || fabs(x[0]-DISPLAYSMENT- WIDTH_2) < 1e-7)
            //     {
            //         f[i * dim + 1] = -500.0/double(RES);
            //     }
                    
            //     else 
            //         f[i * dim + 1] = -1000.0/double(RES);
            //     //std::cout << "add force to node " << i << std::endl;
            // }

            // if (fabs(x[1]-2.) < 1e-7)
            // //if(i >= 1100 && i<1110)
            // {
            //     if (x[0]-DISPLAYSMENT+1e-5 > 1e-7 && x[0]-DISPLAYSMENT-1.5-1e-5 < 1e-7)
            //         continue;
            //     //for (int d = 0; d < dim; d++)
            //     else if (fabs(x[0]- 0) < 1e-7 || fabs(x[0]- 4.) < 1e-7)
            //         f[i * dim + 1] = -1000.0/double(RES);
            //     else {
            //         f[i * dim + 1] = -2000.0/double(RES);
            //         std::cout << "add force to node " << i << std::endl;
            //     }
                    
            // }
        }
    }
    if(!TEST)
    {
        Eigen::VectorXd seg_len(contact_vertices.size()-1);
        int index_3 = 0, index_4 = 0;
        for(int i=0; i<contact_vertices.size()-1; ++i)
        {
            int x = contact_vertices[i];
            int y = contact_vertices[i+1];
            if(true)
            {
                if(x == index_1)
                {
                    index_3 = i;
                }
                else if(y == index_2)
                {
                    index_4 = i;
                }
            }
            double dist = deformed(2*y)-deformed(2*x);
            seg_len(i) = dist;
            //std::cout<<dist<<" ";
        }
        //std::cout<<std::endl;

        //std::cout<<"Index: "<<index_1<<" "<<index_2<<std::endl;

        f[contact_vertices[0] * dim + 1] += -FORCE*seg_len(0)/2;
        if (MODE == 1)
        {
            f[(index_1+1)* dim + 1] += -FORCE*seg_len(index_3)/2;
            f[(index_2-1)* dim + 1] += -FORCE*seg_len(index_4)/2;
        }
        f[contact_vertices[contact_vertices.size()-1] * dim + 1] = -FORCE*seg_len(seg_len.size()-1)/2;

        //std::cout<<" 3,4 "<<index_3<<" "<<index_4<<std::endl;
        for(int i=1; i<contact_vertices.size()-1; ++i)
        {
            if(MODE == 1)
            {
                if(i == index_3 + 1)
                {
                    f[contact_vertices[i] * dim + 1] += -FORCE*(seg_len(i))/2;
                }
                else if (i == index_4)
                {
                    f[contact_vertices[i] * dim + 1] += -FORCE*(seg_len(i-1))/2;
                }
                else
                    f[contact_vertices[i] * dim + 1] += -FORCE*(seg_len(i-1)+seg_len(i))/2;
            }else{
                f[contact_vertices[i] * dim + 1] += -FORCE*(seg_len(i-1)+seg_len(i))/2;
            }
            //std::cout<<i<<" "<<contact_vertices[i]<<std::endl;
                
        }
    }
    for(int i=0; i<f.size(); ++i)
        std::cout<<i/2<<" "<<i%2<<" "<<f[i]<<std::endl;
    if(USE_IMLS)
        findProjectionIMLS(ipc_vertices,true,false);
}

template<int dim>
void FEMSolver<dim>::addNeumannBCFromSTL()
{
    //std::cout<<MassMatrix<<std::endl;
    // std::cout<<GAP<<std::endl;
    f.setZero();
    if(TEST_CASE == 3)
    {
        f[2*254] = FORCE;
    }

    if(USE_SHELL)
    {
        //for(int i=4225; i<num_nodes; ++i)
        for(int i=4225; i<num_nodes; ++i)
        //int i = 433,589,516;
        {
            //  f(3*(557-224+256)+1) = -FORCE;
            //  f(3*(557-224+256)+1) = -FORCE;
            // f(3*(1183-832+288)+1) = -FORCE;
            // f(3*(1165-832+288)+1) = -FORCE;
           f(3*i+1) = -FORCE;
            //f(3*516+1) = -FORCE;
            // f(3*375+1) = -FORCE;
            // f(3*344+1) = -FORCE;
        }
        return;
    }

    std::vector<int> contact_vertices;
    if(dim == 3)
    {
        contact_vertices.resize(force_nodes_3d[0].size());
        for(auto it=force_nodes_3d[0].begin(); it != force_nodes_3d[0].end(); ++it)
        {
            contact_vertices[it->second] = it->first;           
        }
    }
    

    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<dim>(i * dim);
        if(dim == 2)
        {
             if(TEST_CASE == 0)
            {
                
                if((fabs(x[1]-GAP-2.0) < 1e-5))
                    contact_vertices.push_back(i);
                
            }
            else if(TEST_CASE == 1 || TEST_CASE == 2) 
            {
                //contact_vertices = {152,187,191,186,190,185,189,184,188,183,150};
                contact_vertices = force_nodes;
            }
        }
       

    }

    if(contact_vertices.size() > 0)
    {
        if(dim == 2)
        {
            Eigen::VectorXd seg_len(contact_vertices.size()-1);
            for(int i=0; i<contact_vertices.size()-1; ++i)
            {
                int x = contact_vertices[i];
                int y = contact_vertices[i+1];
                double dist = deformed(2*y)-deformed(2*x);
                seg_len(i) = dist;
            }
            f[contact_vertices[0] * dim + 1] += -FORCE*seg_len(0)/2;
            f[contact_vertices[contact_vertices.size()-1] * dim + 1] = -FORCE*seg_len(seg_len.size()-1)/2;

            for(int i=1; i<contact_vertices.size()-1; ++i)
            {
                f[contact_vertices[i] * dim + 1] += -FORCE*(seg_len(i-1)+seg_len(i))/2;
            }
        }

        if(dim == 3)
        {
            double total_force = 0;
            for(int i=0; i<contact_vertices.size(); ++i)
            {
                std::cout<<contact_vertices[i]<<" force: "<<-FORCE*force_nodes_area_3d[0][i]<<std::endl;
                f[contact_vertices[i] * dim + 1] += -FORCE*force_nodes_area_3d[0][i];
                total_force += f[contact_vertices[i] * dim + 1];
            }
            std::cout<<"total force: "<<total_force<<std::endl;
        }
        
    }
    // for(int i=0; i<f.size(); ++i)
    //     std::cout<<i/dim<<" "<<i%dim<<" "<<f[i]<<std::endl;
    
    if(dim == 2)
    {
         if(use_multiple_pairs)
            findProjectionIMLSMultiple(ipc_vertices,true,false);
        else if(USE_IMLS)
            findProjectionIMLS(ipc_vertices,true,false);
    }
    if(dim == 3&& USE_IMLS)
    {
        //findProjectionIMLSMultiple3D(ipc_vertices,true,false);
        // for(int i=0; i<boundary_info.size(); ++i)
        // {
        //     if(boundary_info[i].dist < 0)
        //     {
        //         int index = boundary_info[i].slave_index;
        //         undeformed(dim*index+1) -= boundary_info[i].dist;
        //         deformed(dim*index+1) -= boundary_info[i].dist;

        //         if(IMLS_BOTH && i >= boundary_info_start_3d[0][slave_nodes_3d.size()])
        //         {
        //             undeformed(dim*index+1) += boundary_info[i].dist;
        //             deformed(dim*index+1) += boundary_info[i].dist;
        //         }
        //     }
        // }
    }
   
}


template class FEMSolver<2>;
template class FEMSolver<3>;

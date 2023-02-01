//
// Created by YY C on 2023/01/16.
//

//#include <cstdio>
#include <iostream>
#include <cmath>

#include "config.h"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <omp.h>
#include <chrono>

const float epsilon = 1e-6;

float rand01(){
    return float(rand()) / float(RAND_MAX);
}

int round_up(const float f, const float cell_recpr){
    return int((floor(f * cell_recpr) + 1));
}

float poly6_value(const float s, const float h, const float poly6_factor){
    float result = 0.0;
    if ((0 < s) && (s < h)){
        float x = (h * h - s * s) / (h * h * h);
        result = poly6_factor * x * x * x;
    }
    return result;
}

float spiky_gradient(float dx, float dy, const float h, const float spiky_grad_factor){
    float result = 0;
    float r_len = sqrt(dx*dx + dy*dy);
    if((0 < r_len) && (r_len < h)){
        float x = (h - r_len) / (h * h * h);
        float g_factor = spiky_grad_factor * x * x;
        result = g_factor / r_len;
    }
    return result;
}

float compute_scorr(float dx, float dy, const float corrK, const float corr_deltaQ_coeff, const float h_, const float poly6_factor){
    float norm = sqrt(dx * dx + dy * dy);
    float x = poly6_value(norm, h_, poly6_factor) / poly6_value(corr_deltaQ_coeff * h_,h_,poly6_factor);
    //pow(x, 4)
    x = x * x;
    x = x * x;
    return (-corrK) * x;
}

void confine_position_to_boundary(float &x, float &y, const float* bounding){

    if (x < bounding[0]){
        x = bounding[0] + epsilon * rand01();
    }
    else if(x > bounding[2]){
        x = bounding[2] - epsilon * rand01();
    }

    if (y < bounding[1]){
        y = bounding[1] + epsilon * rand01();
    }
    else if(y > bounding[3]){
        y = bounding[3] - epsilon * rand01();
    }
}


int main() {
    const int screen_res[2] = {4096, 2048};
    const float screen_to_world_ratio = 10.0;
    const float boundary[2] = {screen_res[0] / screen_to_world_ratio, screen_res[1] / screen_to_world_ratio};
    const float cell_size = 2.51;
    const float cell_recpr = 1.0f / cell_size;
    const int grid_size[2] = {round_up(boundary[0], cell_recpr), round_up(boundary[1], cell_recpr)};

    const int num_particles_x = 500;
    const int num_particles = num_particles_x * 60;
    const float time_delta = 1.0 / 20.0;

    const float particle_radius_in_world = 0.25;
    const float particle_radius = particle_radius_in_world * screen_to_world_ratio;

    ///PBF params
    const float h_ = 1.1;
    const float mass = 1.0;
    const float rho0 = 1.0;
    const float lambda_epsilon = 100.0;
    const int pbf_num_iters = 5;
    const float corr_deltaQ_coeff = 0.3;
    const float corrK = 0.001;
    const float neighbor_radius = h_ * 1.05f;

    const float pi = 2 * acos(0.0f);
    const float poly6_factor = 315.0f / 64.0f / pi;
    const float spiky_grad_factor = -45.0f / pi;
    const float g = 3;
    const float v_max = 10000;
    float bounding[4] = {particle_radius_in_world,particle_radius_in_world,
                         boundary[0] - particle_radius_in_world,boundary[1] - particle_radius_in_world};
    const int offset[18] = { -1,-1,   -1, 0,   -1, 1,
                             0,-1,    0, 0,    0, 1,
                             1,-1,    1, 0,    1, 1  };

    ///initialization
    float *x,*y,*old_x,*old_y,*vx,*vy,*lambdas,*position_deltas_x,*position_deltas_y;
    int *grid_count,*grid_trace;
    x = new float[num_particles];
    y = new float[num_particles];
    old_x = new float[num_particles];
    old_y = new float[num_particles];
    vx = new float[num_particles];
    vy = new float[num_particles];
    lambdas = new float[num_particles];
    position_deltas_x = new float[num_particles];
    position_deltas_y = new float[num_particles];
    grid_trace = new int[num_particles];
    grid_count = new int[grid_size[0] * grid_size[1] + 1];

    for(int i = 0; i < num_particles; i++){
        x[i] = (i % num_particles_x) * 2.3f*particle_radius_in_world + particle_radius_in_world;
        y[i] = (i / num_particles_x) * 2.3f*particle_radius_in_world + particle_radius_in_world;
        vx[i] = 0.1f * (rand01() - 0.5f);
        vy[i] = 0.1f * (rand01() - 0.5f);
    }


    int time_step = 0;
    int occupy  = 0;
    auto start = std::chrono::system_clock::now();
    while(time_step < 100) {
        time_step++;

//////////////////////////////////////////////////
/// update start
//////////////////////////////////////////////////

        ///prologue moving board
        bounding[2] = boundary[0] * (0.9f + 0.1f * sin(time_step * pi / 90.f / 5.f)) - particle_radius_in_world;
        ///save old positions
        for(int i = 0; i < num_particles; i++){
            old_x[i] = x[i];
            old_y[i] = y[i];
        }

        ///apply gravity within boundary [eq1-4]
        for(int i = 0; i < num_particles; i++){
            vy[i] -= g * time_delta;
            x[i] += vx[i] * time_delta;
            y[i] += vy[i] * time_delta;
            confine_position_to_boundary(x[i], y[i], bounding);
        }

        ///update grid [eq5-7]
        for(int i = 0; i < grid_size[0]*grid_size[1]; i++) {
            grid_count[i] = 0;
        }
        grid_count[grid_size[0] * grid_size[1]] = num_particles;
//#pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < num_particles; i++){
            ///count the number in each grid (example: 2,2,2,2,...,)
            int grid_ind[2] = {int(x[i] * cell_recpr), int(y[i] * cell_recpr)};
            int index = grid_ind[0] + grid_ind[1] *  grid_size[0];
//#pragma omp atomic
            grid_count[index] += 1; //atomic add
        }
        occupy = 0;
        for(int i = 1; i < grid_size[0]*grid_size[1]; i++) {
            ///establish the range list (example: 2,4,6,8,...,1200)
            if(grid_count[i] > 0){
                occupy += 1;
            }
            grid_count[i] += grid_count[i - 1];
        }
//#pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < num_particles; i++){
            ///put each point in its corresponding range
            ///(the list after update: example: (0,2,4,6,...,1198) the last element is 1200 which is not in the range
            int grid_ind[2] = {int(x[i] * cell_recpr), int(y[i] * cell_recpr)};
            int index = grid_ind[0] + grid_ind[1] *  grid_size[0];
            grid_trace[grid_count[index] - 1] = i;
//#pragma omp atomic
            grid_count[index] -= 1;//atomic add
        }
        ///after this process
        ///the points in grid (i,j) is from grid_count[index(i,j)] to grid_count[index(i,j) + 1] in (grid_x, grid_y)
//        std::cout<<std::endl;
//        for(int i = 0; i < grid_size[0]*grid_size[1]; i++) {
//            if(i%grid_size[0] == 0){std::cout<<std::endl;}
//            std::cout<<grid_count[i];
//        }

        ///substep start [eq8-19] The only part that need omp to accelerate
        ///in normal naive omp settings, this loop takes about 99% of the computing time
        for(int j = 0; j < pbf_num_iters; j++){
            ///compute lambdas [eq8-11]
#pragma omp parallel for schedule(dynamic)
            for(int i = 0; i < num_particles; i++){
                float grad_i[2] = {0.0, 0.0};
                float sum_gradient_sqr = 0.0, density_constraint = 0.0;
                int grid_ind[2] = {int(x[i] * cell_recpr), int(y[i] * cell_recpr)};
                int index;
                ///loop over 9 neighbouring grids
                for(int k = 0; k < 9; k++){
                    int current_ind[2] = {grid_ind[0] + offset[2*k], grid_ind[1] + offset[2*k+1]};
                    if( (current_ind[0] < 0) || (current_ind[0] > grid_size[0] - 1) ||
                        (current_ind[1] < 0) || (current_ind[1] > grid_size[1] - 1) ){
                        //skip
                    } else {
                        index = current_ind[0] + current_ind[1] *  grid_size[0];
                        int start_idx = grid_count[index], end_idx = grid_count[index + 1];
                        for(int l = start_idx; l < end_idx; l++){
                            int index_j = grid_trace[l];
                            float dx = x[i] - x[index_j], dy = y[i] - y[index_j];
                            float norm = sqrt(dx * dx + dy * dy);
                            if((norm > epsilon) && (norm < neighbor_radius) && (i != index_j)){
                                float grad_j = spiky_gradient(dx, dy, h_, spiky_grad_factor);
                                float grad_jx = grad_j * dx, grad_jy = grad_j * dy;
                                grad_i[0] += grad_jx;
                                grad_i[1] += grad_jy;
                                sum_gradient_sqr += (grad_jx * grad_jx + grad_jy * grad_jy);
                                density_constraint += poly6_value(norm, h_, poly6_factor);
                            }
                        }
                    }
                }
                density_constraint = (mass * density_constraint / rho0) - 1.0f;
                sum_gradient_sqr += (grad_i[0] * grad_i[0] + grad_i[1] * grad_i[1]);
                lambdas[i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon);
            }
            ///compute position deltas [eq12-15]
#pragma omp parallel for schedule(dynamic)
            for(int i = 0; i < num_particles; i++) {
                float lambda_i = lambdas[i];
                float pos_delta_i[2] = {0.0, 0.0};
                int grid_ind[2] = {int(x[i] * cell_recpr), int(y[i] * cell_recpr)};
                int index;
                ///loop over 9 neighbouring grids
                for(int k = 0; k < 9; k++){
                    int current_ind[2] = {grid_ind[0] + offset[2*k], grid_ind[1] + offset[2*k+1]};
                    if( (current_ind[0] < 0) || (current_ind[0] > grid_size[0] - 1) ||
                        (current_ind[1] < 0) || (current_ind[1] > grid_size[1] - 1) ){
                        //skip
                    } else {
                        index = current_ind[0] + current_ind[1] * grid_size[0];
                        int start_idx = grid_count[index], end_idx = grid_count[index + 1];
                        for (int l = start_idx; l < end_idx; l++) {
                            int index_j = grid_trace[l];
                            float lambda_j = lambdas[index_j];
                            float dx = x[i] - x[index_j], dy = y[i] - y[index_j];
                            float norm = sqrt(dx * dx + dy * dy);
                            if((norm > epsilon) && (norm < neighbor_radius) && (i != index_j)){
                                float scorr_ij = compute_scorr(dx, dy, corrK, corr_deltaQ_coeff, h_, poly6_factor);
                                float grad_j = spiky_gradient(dx, dy, h_, spiky_grad_factor);
                                float grad_jx = grad_j * dx, grad_jy = grad_j * dy;
                                pos_delta_i[0] += (lambda_i + lambda_j + scorr_ij) * grad_jx;
                                pos_delta_i[1] += (lambda_i + lambda_j + scorr_ij) * grad_jy;
                            }
                        }
                    }
                }
                pos_delta_i[0] = pos_delta_i[0] / rho0;
                pos_delta_i[1] = pos_delta_i[1] / rho0;
                position_deltas_x[i] = pos_delta_i[0];
                position_deltas_y[i] = pos_delta_i[1];
            }
            ///apply position deltas [eq16-18]
            for(int i = 0; i < num_particles; i++) {
                x[i] += position_deltas_x[i];
                y[i] += position_deltas_y[i];
            }
        }
        ///substep end [eq8-19]

        ///epilogue[eq20-24]
        for(int i = 0; i < num_particles; i++) {
            //confine to boundary
            confine_position_to_boundary(x[i], y[i], bounding);
            //update velocities
            vx[i] = fmax(fmin((x[i] - old_x[i]) / time_delta, v_max), -v_max);
            vy[i] = fmax(fmin((y[i] - old_y[i]) / time_delta, v_max), -v_max);
        }

//////////////////////////////////////////////////
/// update end
/// //////////////////////////////////////////////////


#ifdef USE_OPENCV
        //////////////////////////////////////////////////
        /// display start
        cv::Scalar bg_color(65, 47, 17), ball_color(135, 133, 6);
        cv::Mat image(screen_res[1], screen_res[0], CV_8UC3, bg_color);
        for (int i = 0; i < num_particles; i++) {
            cv::Point current(x[i] * screen_to_world_ratio, screen_res[1] - y[i] * screen_to_world_ratio);
            cv::circle(image, current, particle_radius, ball_color, -1, cv::LINE_AA);
        }
        for (int i = 0; i < grid_size[0]; i++) {
            cv::Point pt1(i * cell_size * screen_to_world_ratio,0), pt2(i * cell_size * screen_to_world_ratio,screen_res[1]);
            cv::line(image, pt1, pt2, ball_color, 1, cv::LINE_AA);
        }
        for (int i = 0; i < grid_size[1]; i++) {
            cv::Point pt1(0, i * cell_size * screen_to_world_ratio), pt2(screen_res[0], i * cell_size * screen_to_world_ratio);
            cv::line(image, pt1, pt2, ball_color, 1, cv::LINE_AA);
        }

        cv::imshow("result", image);
        if (cv::waitKey(1) == 27) {
            break;
        }
        /// display end
        //////////////////////////////////////////////////
#endif
    }

    //////////////////////////////////////////////////
    /// time
    //////////////////////////////////////////////////
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout<<"time = "<<double(duration.count())<<"ms"<<std::endl;
    printf("%f\n", num_particles/float(occupy));


    return 0;
}
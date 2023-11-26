#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stack>
#include <limits>
#include <queue>
#include <algorithm>
#include <execution>
#include <numeric>
#include <fstream>
#include <ppl.h>
#include <sstream>
using namespace std;
using namespace cv;

extern vector<Mat> warped_imgs, comps, Fecker_comps, cites_range, wave_num;
extern vector<vector<Mat>> masks, overlaps, seams;
extern vector<Point> seam_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst, sorted_seam_pixel_lst, target_pixel_lst;
extern vector<vector<int>> ref_seq;
extern Mat result_img, anomaly_mask, discovered_map, range_map, discovered_time_stamp_map, color_comp_map, Fecker_color_comp_map, range_count_map;
extern vector<int> exe_seq;
extern vector<bool> check_corrected;

// super parameters
extern double cost_threshold, min_sigma_color, sigma_color, sigma_dist, correlation_TH, Fecker_enhance;
extern int propagation_coefficient, height, width, correlation_block, corr_status, exe_seq_sw;
extern bool refercon;

struct ImgPack
{
public:
    vector<float> PDF;
    vector<float> CDF;
};

namespace Utils
{
    // generate a list consists of points on stitching line.
    vector<Point> build_seam_pixel_lst(Mat& seam_mask);
    bool refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst, vector<Point>& discarded_seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img);
    void sort_seam_pixel_lst(vector<Point>& seam_pixel_lst, vector<Point>& sorted_seam_pixel_lst, Mat& seam_mask);
    void build_range_map_with_side_addition(vector<Point>& sorted_seam_pixel_lst, Mat& result_from_tar_mask, Mat& tar_img);
    vector<Point> build_target_pixel_lst(Mat& result_from_tar_mask, Mat& seam_mask, Mat& tar_img);
    void init_color_comp_map(vector<Point>& seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img);

    //void update_color_comp_map_range_anomaly(vector<vector<Point>>& sup_pxl_lst, Mat& warped_tar_img);
    //vector<double> get_color_comp_value_range_anomaly(int x, int y, Mat& warped_tar_img);
    void update_color_comp_map_range_anomaly__parallel_ver(vector<vector<Point>>& sup_pxl_lst, Mat& warped_tar_img);
    vector<double> get_color_comp_value_range_anomaly__parallel_ver(int x, int y, Mat& warped_tar_img);

    //Fecker func
    ImgPack CalDF(Mat& src, int channel, Mat& overlap);
    Mat HM_Fecker(Mat& ref, Mat& tar, Mat& overlap, Mat& result_from_tar_mask);
    Mat HM_Fecker_corr(Mat& ref, Mat& tar, Mat& overlap, Mat& result_from_tar_mask);
    vector<Mat> correlation_coef_estimate(Mat& ref, Mat& tar, Mat& overlap);
    vector<Mat> correlation_cos_estimate(Mat& ref, Mat& tar, Mat& overlap);
    vector<Mat> correlation_cos2_estimate(Mat& ref, Mat& tar, Mat& overlap);

    //ordering algorithm
    double Thenengrad(Mat& img);
    double laplacian(Mat& img);

    //Proposed fusion color correction method and compared method
    void EF_based();

    //hole fulling
    Mat hole_detection();
    void hole_filling(Mat& hole);
    void hole_filling_ver2(Mat& hole);
    void hole_filling_ver3(Mat& hole);

    // build result
    void build_final_result_Before(Mat& warped_tar_img, int num);
    void build_final_result_Before_with_Fecker_weighting(Mat& warped_tar_img, int num);
    Mat build_final_result();
}

namespace Inits
{
    // load data.
    vector<string> getimglist(string& addr);
    vector<Mat> Loadimage(string& addr, int flag);
    vector<vector<Mat>> Loadmask(string& addr, int N);
    vector<vector<Mat>> Loadmask_ver2(string& addr, int N);
    void loadAll(string& warpDir, string& seamDir, string& overlapDir, string& maskDir);
    void build_exe_sequence();

    //initialization
    void Initialize_var();
    vector<int> refimg_num(int ignore, int img_amount);
}
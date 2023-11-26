#include "utili.h"
vector<Mat> warped_imgs, comps, Fecker_comps, cites_range, wave_num;
vector<vector<Mat>> masks, overlaps, seams;
vector<Point> seam_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst, sorted_seam_pixel_lst, target_pixel_lst;
vector<vector<int>> ref_seq;
Mat result_img, anomaly_mask, discovered_map, range_map, discovered_time_stamp_map, color_comp_map, Fecker_color_comp_map, range_count_map;
vector<int> exe_seq;
vector<bool> check_corrected;

// parameters
double cost_threshold, min_sigma_color, sigma_color, sigma_dist, correlation_TH, Fecker_enhance;
int propagation_coefficient, height, width, correlation_block, corr_status, exe_seq_sw;
bool refercon;

int main()
{
    // parameters setting
    corr_status = 1; // 0.none; 1.correlation_coef; 2.cos(mean, sd); 3.cos(every_pixel_color_intensity);
    correlation_block = 2;
    correlation_TH = 0.6;
    Fecker_enhance = -0.3;
    exe_seq_sw = 1; // 1.decreasing(>); 2.increasing(<);

    sigma_dist = 10;
    sigma_color = 10;
    min_sigma_color = 0.5;
    cost_threshold = 500;
    propagation_coefficient = 2;

    cout << "# Running CoColour ...\n\n";

    string data_num = "data/test_dataset/01";
    string warpDir = data_num + "/multiview_warp_result/";

    string maskDir = data_num + "/mask/";
    string overlapDir = data_num + "/overlap/";
    string seamDir = data_num + "/seam/";

    Inits::loadAll(warpDir, seamDir, overlapDir, maskDir);
    height = warped_imgs[0].rows;
    width = warped_imgs[0].cols;

    Inits::Initialize_var();

    clock_t start_time = clock();
    ;
    Inits::build_exe_sequence();
    for (auto &item : exe_seq)
        cout << item << " ";

    Utils::EF_based();

    result_img = Utils::build_final_result();
    cout << "execution time: " << double(clock() - start_time) / CLOCKS_PER_SEC << "s." << endl;

    // save result
    for (int i = 0; i < warped_imgs.size(); i++)
    {
        string str = to_string(i) + "__warp.png";
        imwrite("data/" + str, warped_imgs[i]);
    }
    Mat hole = Utils::hole_detection();
    Utils::hole_filling_ver3(hole);
    imwrite("data/final_result.png", result_img);

    cout << warpDir << "done!!\n";
}
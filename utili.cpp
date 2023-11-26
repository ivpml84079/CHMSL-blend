#include "utili.h"

float clip(float n, float lower, float upper)
{
    n = (n > lower) * n + !(n > lower) * lower;
    return (n < upper) * n + !(n < upper) * upper;
}

vector<string> Inits::getimglist(string &addr)
{
    ofstream file1("files.csv");
    file1.trunc;
    file1 << addr << ",";
    file1.close();

    system("python .\\needPYhelp.py");

    vector<string> imgs;
    fstream file;
    file.open("files.csv");
    string S, T;
    cout << "getimglist---->";
    while (getline(file, S))
    {
        istringstream X(S);
        while (getline(X, T, ','))
        {
            imgs.push_back(T);
        }
    }
    file.close();
    cout << "done!\n\n";

    return imgs;
}

vector<Mat> Inits::Loadimage(string &addr, int flag)
{
    vector<Mat> imgs;
    vector<string> img_name = Inits::getimglist(addr);
    cout << "Load " << img_name.size() << " images---->";

    for (int i = 0; i < img_name.size(); i++)
    {
        // cout << img_name[i] << endl;
        imgs.push_back(imread(addr + img_name[i], flag));
    }
    cout << "done!\n\n";
    return imgs;
}

vector<vector<Mat>> Inits::Loadmask(string &addr, int N)
{
    cout << "N = " << N << endl;
    vector<vector<Mat>> compilation;
    vector<string> img_name = Inits::getimglist(addr);
    cout << "Load " << img_name.size() << " images---->";

    for (int i = 0; i < N; i++)
    {
        vector<Mat> imgs;
        for (int j = (N - 1) * i; j < (N - 1) * i + N - 1; j++)
        {
            // cout << img_name[j] << endl;
            imgs.push_back(imread(addr + img_name[j], 0));
        }
        compilation.push_back(imgs);
    }

    cout << "done!\n\n";
    return compilation;
}

vector<vector<Mat>> Inits::Loadmask_ver2(string &addr, int N)
{
    cout << "N = " << N << endl;
    vector<vector<int>> seq(N);
    vector<vector<Mat>> compilation(N);
    vector<string> img_name = Inits::getimglist(addr);
    cout << "Load " << img_name.size() << " images---->";

    for (int i = 0; i < img_name.size(); i++)
    {
        int num = img_name[i][0] - '0';
        /*cout << "real1: " << num << endl;
        cout << "real2: " << img_name[i][3] - '0' << endl;*/
        seq[num].push_back(img_name[i][3] - '0');
        compilation[num].push_back(imread(addr + img_name[i], 0));
    }

    cout << "done!\n\n";
    ref_seq = seq;
    return compilation;
}

void Inits::loadAll(string &warpDir, string &seamDir, string &overlapDir, string &maskDir)
{
    cout << "load warp images-------------------->\n\n";
    warped_imgs = Inits::Loadimage(warpDir, 1);

    int N = warped_imgs.size();

    cout << "load seams-------------------->\n\n";
    // seams = Inits::Loadmask(seamDir, N);
    seams = Inits::Loadmask_ver2(seamDir, N);
    cout << "load overlaps-------------------->\n\n";
    overlaps = Inits::Loadmask_ver2(overlapDir, N);
    cout << "load masks-------------------->\n\n";
    masks = Inits::Loadmask_ver2(maskDir, N);
}

vector<Point> Utils::build_seam_pixel_lst(Mat &seam_mask)
{
    vector<Point> locations;
    cv::findNonZero(seam_mask, locations);
    return locations;
}

bool Utils::refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(vector<Point> seam_pixel_lst, vector<Point> &refined_seam_pixel_lst, vector<Point> &discarded_seam_pixel_lst, Mat &warped_ref_img, Mat &warped_tar_img)
{
    // seam_mask.copyTo(anomaly_map);

    int count = seam_pixel_lst.size();
    Mat points1(count, 1, CV_8U);
    Mat points2(count, 1, CV_8U);
    Mat points3(count, 1, CV_8U);
    Mat label1, label2, label3;
    int centers = 2;
    vector<vector<float>> out1(centers), out2(centers), out3(centers);

    int anomaly_count = 0;
    int ordinary_count = 0;

    for (int i = 0; i < count; i++)
    {
        Point p = seam_pixel_lst[i];

        int distance1 = (int)warped_ref_img.at<Vec3b>(p)[0] - (int)warped_tar_img.at<Vec3b>(p)[0];
        distance1 = abs(distance1);
        int distance2 = (int)warped_ref_img.at<Vec3b>(p)[1] - (int)warped_tar_img.at<Vec3b>(p)[1];
        distance2 = abs(distance2);
        int distance3 = (int)warped_ref_img.at<Vec3b>(p)[2] - (int)warped_tar_img.at<Vec3b>(p)[2];
        distance3 = abs(distance3);

        points1.at<uchar>(i, 0) = (uchar)distance1;
        points2.at<uchar>(i, 0) = (uchar)distance2;
        points3.at<uchar>(i, 0) = (uchar)distance3;
    }

    int T_1 = threshold(points1, label1, 0, 255, THRESH_BINARY | THRESH_OTSU);
    int T_2 = threshold(points2, label2, 0, 255, THRESH_BINARY | THRESH_OTSU);
    int T_3 = threshold(points3, label3, 0, 255, THRESH_BINARY | THRESH_OTSU);

    cout << "OTSU----->\n";

    int sum1 = sum(label1)[0] / 255;
    int sum2 = sum(label2)[0] / 255;
    int sum3 = sum(label3)[0] / 255;
    int target1 = sum1 < count / 2 ? 0 : 255;
    int target2 = sum2 < count / 2 ? 0 : 255;
    int target3 = sum3 < count / 2 ? 0 : 255;

    for (int i = 0; i < count; i++)
    {
        if (label1.at<uchar>(i) == target1 && label2.at<uchar>(i) == target2 && label3.at<uchar>(i) == target3)
        {
            refined_seam_pixel_lst.push_back(seam_pixel_lst[i]);
        }
        else
        {
            anomaly_count++;
            Point p = seam_pixel_lst[i];
            // anomaly_map.at<uchar>(p) = 0;
            discarded_seam_pixel_lst.push_back(seam_pixel_lst[i]);
        }

        // new shit
        if (points1.at<uchar>(i, 0) < T_1)
            out1[0].push_back(points1.at<uchar>(i, 0));
        else
            out1[1].push_back(points1.at<uchar>(i, 0));

        if (points2.at<uchar>(i, 0) < T_2)
            out2[0].push_back(points2.at<uchar>(i, 0));
        else
            out2[1].push_back(points2.at<uchar>(i, 0));

        if (points3.at<uchar>(i, 0) < T_3)
            out3[0].push_back(points3.at<uchar>(i, 0));
        else
            out3[1].push_back(points3.at<uchar>(i, 0));
    }

    ordinary_count = count - anomaly_count;
    double cost1 = double(ordinary_count * anomaly_count) * pow(sum(out1[0])[0] / out1[0].size() - sum(out1[1])[0] / out1[1].size(), 2) / double(count * count);
    double cost2 = double(ordinary_count * anomaly_count) * pow(sum(out2[0])[0] / out2[0].size() - sum(out2[1])[0] / out2[1].size(), 2) / double(count * count);
    double cost3 = double(ordinary_count * anomaly_count) * pow(sum(out3[0])[0] / out3[0].size() - sum(out3[1])[0] / out3[1].size(), 2) / double(count * count);

    if (out1[0].size() == 0)
        cost1 = 0.1;
    if (out2[0].size() == 0)
        cost2 = 0.1;
    if (out3[0].size() == 0)
        cost3 = 0.1;

    cout << "test: " << sum(out3[0])[0] << ", " << sum(out3[1])[0] << endl;

    cout << "The cost1 is " << cost1 << endl;
    cout << "The cost2 is " << cost2 << endl;
    cout << "The cost3 is " << cost3 << endl;

    cout << "\nmisalignment rate: " << round((float)anomaly_count / count * 100) / 100 << endl;
    cout << "\nmisalignment count: " << anomaly_count << endl;

    if (cost1 < cost_threshold && cost2 < cost_threshold && cost3 < cost_threshold)
    {
        refined_seam_pixel_lst = seam_pixel_lst;
        discarded_seam_pixel_lst.clear();
        return true;
    }
    else
    {
        return false;
    }
}

void Utils::sort_seam_pixel_lst(vector<Point> &seam_pixel_lst, vector<Point> &sorted_seam_pixel_lst, Mat &seam_mask)
{
    Mat seam_discover_map;
    seam_mask.copyTo(seam_discover_map);
    vector<Point> endpoints_lst;
    int x_off[4] = {-1, 1, 0, 0};
    int y_off[4] = {0, 0, -1, 1};
    for (Point p : seam_pixel_lst)
    {
        int x = p.x;
        int y = p.y;
        int count = 0;

        for (int i = 0; i < 4; i++)
        {
            int nx = x + x_off[i];
            int ny = y + y_off[i];

            if (seam_mask.at<uchar>(ny, nx) == 255)
                count++;
        }
        if (count == 1)
        {
            endpoints_lst.push_back(p);
        }
    }

    stack<Point> discover_stack;
    discover_stack.push(endpoints_lst[0]);
    seam_discover_map.at<uchar>(endpoints_lst[0]) = 0;
    sorted_seam_pixel_lst.push_back(endpoints_lst[0]);

    while (!discover_stack.empty())
    {
        Point p = discover_stack.top();
        discover_stack.pop();
        int x = p.x;
        int y = p.y;

        for (int i = 0; i < 4; i++)
        {
            int nx = x + x_off[i];
            int ny = y + y_off[i];

            if (seam_discover_map.at<uchar>(ny, nx) == 255)
            {
                Point new_p = Point(nx, ny);
                seam_discover_map.at<uchar>(new_p) = 0;
                sorted_seam_pixel_lst.push_back(new_p);
                discover_stack.push(new_p);
            }
        }
    }
}

void Utils::build_range_map_with_side_addition(vector<Point> &sorted_seam_pixel_lst, Mat &result_from_tar_mask, Mat &tar_img)
{
    range_count_map = Mat::zeros(height, width, CV_32FC1);
    for (int idx = 0; idx < sorted_seam_pixel_lst.size(); idx++)
    {
        range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[0] = idx;
        range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[1] = idx;
    }

    int max_cite_range = sorted_seam_pixel_lst.size() - 1;
    int time_stamp = 1;
    static uint8_t gray = 255;
    static int x_offset[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static int y_offset[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    bool flag_early_termination = false;
    vector<Point> next_wavefront;
    next_wavefront.reserve(10000);
    vector<Point> current_wavefront = sorted_seam_pixel_lst;

    for (Point point : sorted_seam_pixel_lst)
        discovered_map.at<uchar>(point) = 255;
    for (Point point : sorted_seam_pixel_lst)
        discovered_time_stamp_map.at<ushort>(point) = time_stamp;

    clock_t start, end;
    double cpu_time_used;

    // march through whole target image.
    while (true)
    {
        // find next wavefront
        time_stamp++;
        next_wavefront.clear();
        for (Point point : current_wavefront)
        {
            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n == width || y_n == height)
                    continue;
                // bool is_not_discovered = discovered_map.at<uchar>(y_n, x_n) == 0 && result_from_tar_mask.at<uchar>(y_n, x_n) == 255;
                // �h�[����
                bool sw = tar_img.at<Vec3b>(y_n, x_n)[0] != 0 || tar_img.at<Vec3b>(y_n, x_n)[1] != 0 || tar_img.at<Vec3b>(y_n, x_n)[2] != 0;
                bool is_not_discovered = discovered_map.at<uchar>(y_n, x_n) == 0 && sw;
                // �h�[����
                if (is_not_discovered)
                {
                    next_wavefront.push_back(Point(x_n, y_n));
                    discovered_map.at<uchar>(y_n, x_n) = 255;
                    discovered_time_stamp_map.at<ushort>(y_n, x_n) = time_stamp;
                    // test_wavefront_marching.at<uchar>(y_n, x_n) = gray;
                }
            }
        }

        // break from the while loop if there is no next wavefront.
        if (next_wavefront.size() == 0)
            break;
        int reference_all_range_cnt = 0;
        // propagate the citation range from previous wavefront to current wavefront.
        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            if (!flag_early_termination)
            {
                for (int i = 0; i < 8; i++)
                {
                    int y_n = point.y + y_offset[i];
                    int x_n = point.x + x_offset[i];
                    if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                        continue;
                    if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp - 1)
                    {
                        if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                            min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                        if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                            max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                    }
                }

                min_range = max(min_range - propagation_coefficient, 0);
                max_range = min(max_range + propagation_coefficient, max_cite_range);
                range_map.at<Vec2w>(point)[0] = min_range;
                range_map.at<Vec2w>(point)[1] = max_range;

                if (range_map.at<Vec2w>(point)[0] == 0 && range_map.at<Vec2w>(point)[1] == max_cite_range)
                {
                    // Fecker_mask.at<uchar>(point) = 255;
                    reference_all_range_cnt++;
                }

                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
            else
            {
                range_map.at<Vec2w>(point)[0] = 0;
                range_map.at<Vec2w>(point)[1] = max_cite_range;
                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
        }
        if (next_wavefront.size() == reference_all_range_cnt)
            flag_early_termination = true;

        vector<int> pending_update_min_ranges, pending_update_max_ranges;

        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                    continue;
                if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp)
                {
                    if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                        min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                    if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                        max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                }
            }
            pending_update_min_ranges.push_back(min_range);
            pending_update_max_ranges.push_back(max_range);
        }

        for (int idx = 0; idx < next_wavefront.size(); idx++)
        {
            range_map.at<Vec2w>(next_wavefront[idx])[0] = pending_update_min_ranges[idx];
            range_map.at<Vec2w>(next_wavefront[idx])[1] = pending_update_max_ranges[idx];
        }

        gray -= 10;

        current_wavefront = next_wavefront;
    }
}

vector<Point> Utils::build_target_pixel_lst(Mat &result_from_tar_mask, Mat &seam_mask, Mat &tar_img)
{
    vector<Point> locations;
    // cv::findNonZero(result_from_tar_mask - seam_mask, locations);

    // �h�[����
    Mat mask_test;
    tar_img.copyTo(mask_test);
    cvtColor(mask_test, mask_test, COLOR_RGB2GRAY);
    // imwrite("data/mask_test.png", mask_test-seam_mask);
    // cv::findNonZero(mask_test - seam_mask, locations);
    cv::findNonZero(mask_test, locations);
    // �h�[����

    return locations;
}

void Utils::init_color_comp_map(vector<Point> &seam_pixel_lst, Mat &warped_ref_img, Mat &warped_tar_img)
{
    for (int i = 0; i < seam_pixel_lst.size(); i++)
    {
        int x = seam_pixel_lst[i].x;
        int y = seam_pixel_lst[i].y;
        color_comp_map.at<Vec3d>(y, x)[0] = double(warped_ref_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y, x)[0]);
        color_comp_map.at<Vec3d>(y, x)[1] = double(warped_ref_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y, x)[1]);
        color_comp_map.at<Vec3d>(y, x)[2] = double(warped_ref_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y, x)[2]);
    }
}

/*
void Utils::update_color_comp_map_range_anomaly(vector<vector<Point>>& sup_pxl_lst, Mat& warped_tar_img)
{
    for (vector<Point> lst : sup_pxl_lst)
    {
        int x = lst[0].x;
        int y = lst[0].y;
        vector<double> color_comp = get_color_comp_value_range_anomaly(x, y, warped_tar_img);
        color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
        color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
        color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
    }
    for (vector<Point> lst : sup_pxl_lst)
    {
        int x = lst[0].x;
        int y = lst[0].y;
        double aa = color_comp_map.at<Vec3d>(y, x)[0];
        double bb = color_comp_map.at<Vec3d>(y, x)[1];
        double cc = color_comp_map.at<Vec3d>(y, x)[2];
        for (Point pt : lst)
        {
            int x = pt.x;
            int y = pt.y;
            color_comp_map.at<Vec3d>(y, x)[0] = aa;
            color_comp_map.at<Vec3d>(y, x)[1] = bb;
            color_comp_map.at<Vec3d>(y, x)[2] = cc;
        }
    }
}

vector<double> Utils::get_color_comp_value_range_anomaly(int x, int y, Mat& warped_tar_img)
{
    vector<double> color_comp = { 0.0, 0.0, 0.0 };
    double weight_sum = 0.0;
    double a, b;
    int low_bound = range_map.at<Vec2w>(y, x)[0];
    int high_bound = range_map.at<Vec2w>(y, x)[1];
    int range_num = high_bound - low_bound + 1;

    for (int i = 0; i < range_num; i++)
    {
        int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
        int y_seam = sorted_seam_pixel_lst[i + low_bound].y;

        double weight;
        // double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
        double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2) + pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2) + pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

        a = color_diff / sigma_color / sigma_color * -1.0;
        // b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
        // weight = exp(a) * exp(b);
        weight = exp(a);

        color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
        color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
        color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

        weight_sum += weight;
    }
    if (weight_sum == 0.0)
    {
        color_comp[0] = 0.0;
        color_comp[1] = 0.0;
        color_comp[2] = 0.0;
    }
    else
    {
        color_comp[0] = color_comp[0] / weight_sum;
        color_comp[1] = color_comp[1] / weight_sum;
        color_comp[2] = color_comp[2] / weight_sum;
    }

    return color_comp;
}
*/

void Utils::update_color_comp_map_range_anomaly__parallel_ver(vector<vector<Point>> &sup_pxl_lst, Mat &warped_tar_img)
{
    std::for_each(
        std::execution::par,
        sup_pxl_lst.begin(),
        sup_pxl_lst.end(),
        [&](vector<Point> lst)
        {
            int x = lst[0].x;
            int y = lst[0].y;
            vector<double> color_comp = get_color_comp_value_range_anomaly__parallel_ver(x, y, warped_tar_img);
            color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
            color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
            color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
        });

    for (vector<Point> lst : sup_pxl_lst)
    {
        int x = lst[0].x;
        int y = lst[0].y;
        double aa = color_comp_map.at<Vec3d>(y, x)[0];
        double bb = color_comp_map.at<Vec3d>(y, x)[1];
        double cc = color_comp_map.at<Vec3d>(y, x)[2];
        for (Point pt : lst)
        {
            int x = pt.x;
            int y = pt.y;
            color_comp_map.at<Vec3d>(y, x)[0] = aa;
            color_comp_map.at<Vec3d>(y, x)[1] = bb;
            color_comp_map.at<Vec3d>(y, x)[2] = cc;
        }
    }
}

vector<double> Utils::get_color_comp_value_range_anomaly__parallel_ver(int x, int y, Mat &warped_tar_img)
{
    vector<double> color_comp = {0.0, 0.0, 0.0};
    double weight_sum = 0.0;
    double a, b;
    int low_bound = range_map.at<Vec2w>(y, x)[0];
    int high_bound = range_map.at<Vec2w>(y, x)[1];
    int range_num = high_bound - low_bound + 1;
    int total_anomaly_cnt = 0;
    // int max_anomaly_cnt = discarded_seam_pixel_lst.size();

    for (int i = 0; i < range_num; i++)
    {
        int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
        int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
        if (anomaly_mask.at<uchar>(y_seam, x_seam) == 255)
            total_anomaly_cnt++;
    }

    double anomaly_ratio = (double)total_anomaly_cnt / (double)range_num;

    for (int i = 0; i < range_num; i++)
    {
        int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
        int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
        double weight;
        double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
        double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2) + pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2) + pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;
        double new_color_sigma = max(sigma_color * anomaly_ratio, min_sigma_color);
        a = color_diff / new_color_sigma / new_color_sigma * -1.0;
        b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
        weight = exp(a) * exp(b);
        color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
        color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
        color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;
        weight_sum += weight;
    }

    if (weight_sum == 0.0)
    {
        color_comp[0] = 0.0;
        color_comp[1] = 0.0;
        color_comp[2] = 0.0;
    }
    else
    {
        color_comp[0] = color_comp[0] / weight_sum;
        color_comp[1] = color_comp[1] / weight_sum;
        color_comp[2] = color_comp[2] / weight_sum;
    }

    return color_comp;
}

void Inits::Initialize_var()
{
    // initialize map
    range_map = Mat::zeros(height, width, CV_16UC2);
    discovered_time_stamp_map = Mat::zeros(height, width, CV_16U);
    discovered_map = Mat::zeros(height, width, CV_8U);
    color_comp_map = Mat::zeros(height, width, CV_64FC3);
    anomaly_mask = Mat::zeros(height, width, CV_8U);
    seam_pixel_lst.erase(seam_pixel_lst.begin(), seam_pixel_lst.end());
    refined_seam_pixel_lst.erase(refined_seam_pixel_lst.begin(), refined_seam_pixel_lst.end());
    discarded_seam_pixel_lst.erase(discarded_seam_pixel_lst.begin(), discarded_seam_pixel_lst.end());
    sorted_seam_pixel_lst.erase(sorted_seam_pixel_lst.begin(), sorted_seam_pixel_lst.end());
}

vector<int> Inits::refimg_num(int ignore, int img_amount)
{
    vector<int> result;
    for (int i = 0; i < img_amount; i++)
    {
        if (i == ignore)
            continue;
        result.push_back(i);
    }
    return result;
}

void Utils::build_final_result_Before(Mat &warped_tar_img, int num)
{
    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);
    cout << "comps.size: " << comps.size() << endl;
    for (int i = 0; i < comps.size(); i++)
    {
        warped_tar_img = warped_tar_img + comps[i] / comps.size();
    }
    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
}

void Utils::build_final_result_Before_with_Fecker_weighting(Mat &warped_tar_img, int num)
{
    Mat total_wave = Mat::zeros(wave_num[0].size(), wave_num[0].type());
    for (int i = 0; i < wave_num.size(); i++)
        total_wave += wave_num[i];

    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);
    cout << "Fecker_comps.size: " << Fecker_comps.size() << endl;

    int n_1 = sorted_seam_pixel_lst.size();
    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < width; y++)
        {
            double n_2 = 0.0, n_2_normalized = 0.0;
            for (int i = 0; i < Fecker_comps.size(); i++)
            {
                if (masks[num][i].at<bool>(x, y) == false)
                    continue;
                n_2 += wave_num[i].at<ushort>(x, y);
            }

            for (int i = 0; i < Fecker_comps.size(); i++)
            {
                if (masks[num][i].at<bool>(x, y) == false)
                    continue;
                n_2_normalized += n_2 / wave_num[i].at<ushort>(x, y);
            }

            if (n_2 == 0)
                continue;
            for (int i = 0; i < Fecker_comps.size(); i++)
            {
                if (masks[num][i].at<bool>(x, y) == false)
                    continue;

                for (int c = 0; c < 3; c++)
                {
                    double rate = 1.0 * cites_range[i].at<int>(x, y) / n_1;
                    rate = (Fecker_comps[i].at<Vec3d>(x, y)[c] * clip(rate + Fecker_enhance, 0, 1)) + (comps[i].at<Vec3d>(x, y)[c] * clip(1 - rate - Fecker_enhance, 0, 1));
                    warped_tar_img.at<Vec3d>(x, y)[c] += rate * (n_2 / (wave_num[i].at<ushort>(x, y)) / n_2_normalized);
                }
            }
        }
    }
    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
}

Mat Utils::build_final_result()
{
    Mat result, mask_tmp;
    int N = warped_imgs.size();

    for (int i = 0; i < N; i++)
    {
        masks[i][0].copyTo(mask_tmp);
        for (int j = 1; j < masks[i].size(); j++)
        {
            bitwise_and(mask_tmp, masks[i][j], mask_tmp);
        }
        warped_imgs[i].copyTo(result, mask_tmp);
    }

    return result;
}

Mat Utils::hole_detection()
{
    Mat hole = Mat(masks[0][0].size(), masks[0][0].type());
    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < width; y++)
        {
            if (result_img.at<Vec3b>(x, y)[0] == 0 && result_img.at<Vec3b>(x, y)[1] == 0 && result_img.at<Vec3b>(x, y)[2] == 0)
                hole.at<uchar>(x, y) = 255;
        }
    }

    return hole;
}

void Utils::hole_filling(Mat &hole)
{
    Mat filling = Mat(warped_imgs[0].size(), warped_imgs[0].type());
    for (int i = 0; i < warped_imgs.size(); i++)
    {
        filling += warped_imgs[i] / warped_imgs.size();
    }
    filling.copyTo(result_img, hole);
}

void Utils::hole_filling_ver2(Mat &hole)
{
    Mat filling = Mat(warped_imgs[0].size(), warped_imgs[0].type());
    vector<Mat> Mat_temp;
    int count = 0;
    for (int i = 0; i < warped_imgs.size(); i++)
    {
        Mat mask_temp = masks[i][0];
        vector<Point> check_zero;
        for (int j = 1; j < masks[i].size(); j++)
        {
            bitwise_and(mask_temp, masks[i][j], mask_temp);
        }
        bitwise_and(mask_temp, hole, mask_temp);
        findNonZero(mask_temp, check_zero);
        if (check_zero.size() == 0)
            continue;
        count++;
        Mat_temp.push_back(warped_imgs[i]);
    }
    for (int i = 0; i < Mat_temp.size(); i++)
    {
        filling += Mat_temp[i] / count;
    }

    filling.copyTo(result_img, hole);
}

void Utils::hole_filling_ver3(Mat &hole)
{
    Mat filling = Mat(warped_imgs[0].size(), warped_imgs[0].type());
    vector<Point> location;
    findNonZero(hole, location);
    for (auto &loc : location)
    {
        vector<double> color(3, 0.0);
        int count = 0;
        int thres = 10;
        for (int i = 0; i < warped_imgs.size(); i++)
        {
            /*if (warped_imgs[i].at<Vec3b>(loc.y, loc.x)[0] == 0 && warped_imgs[i].at<Vec3b>(loc.y, loc.x)[1] == 0 && warped_imgs[i].at<Vec3b>(loc.y, loc.x)[2] == 0)
            {
                continue;
            }*/
            if (warped_imgs[i].at<Vec3b>(loc.y, loc.x)[0] < thres && warped_imgs[i].at<Vec3b>(loc.y, loc.x)[1] < thres && warped_imgs[i].at<Vec3b>(loc.y, loc.x)[2] < thres)
            {
                continue;
            }
            color[0] += warped_imgs[i].at<Vec3b>(loc.y, loc.x)[0];
            color[1] += warped_imgs[i].at<Vec3b>(loc.y, loc.x)[1];
            color[2] += warped_imgs[i].at<Vec3b>(loc.y, loc.x)[2];
            count++;
        }
        for (int i = 0; i < color.size(); i++)
        {
            color[i] /= count;
            filling.at<Vec3b>(loc.y, loc.x)[i] = color[i];
        }
    }

    filling.copyTo(result_img, hole);
}

// Fecker func
ImgPack Utils::CalDF(Mat &src, int channel, Mat &overlap)
{
    ImgPack src_DF;
    src_DF.PDF.assign(256, 0);
    src_DF.CDF.assign(256, 0);

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (overlap.at<bool>(i, j) == true)
                src_DF.PDF[(int)src.at<Vec3b>(i, j)[channel]]++;

    src_DF.CDF[0] = src_DF.PDF[0];

    for (int i = 1; i < src_DF.PDF.size(); i++)
        src_DF.CDF[i] = src_DF.PDF[i] + src_DF.CDF[i - 1];

    return src_DF;
}

Mat Utils::HM_Fecker(Mat &ref, Mat &tar, Mat &overlap, Mat &tar_mask)
{
    vector<int> mapping_Func[3];

    for (int channel = 0; channel < 3; channel++)
    {
        mapping_Func[channel].assign(256, 0);

        ImgPack ref_DF, tar_DF;
        ref_DF = CalDF(ref, channel, overlap);
        tar_DF = CalDF(tar, channel, overlap);

        // �u�n�O���Ĥ@�ӡA�ҥH�ݭn�@�Ӷ}���A�����Ĥ@�ӥ��`�M�g�Υ��`�Q�M�g����
        bool flag = false;
        int temp_x = -100, temp_y = -100;

        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                if (ref_DF.CDF[j] > tar_DF.CDF[i])
                {
                    if (flag == false && (j - 1) >= 0)
                    {
                        flag = true; // �}�����}!
                        temp_x = i;
                        temp_y = j;
                    }

                    // saturate_cast<uchar>�i�H�T�O�ƭȦbuchar 8bits��(0-255).
                    mapping_Func[channel][i] = (int)saturate_cast<uchar>(j);
                    break;
                }

                // ���L�ȥi�M�g�ɨ��e�@�ӬM�g��
                mapping_Func[channel][i] = mapping_Func[channel][i - 1];
            }
        }

        // cout << "tempx,y: " << temp_x << ' ' << temp_y << endl;

        // ���L�ȥi�M�g�ɨ��Ĥ@�ӥ��`�M�g����
        for (int i = temp_x; i >= 0; i--)
            mapping_Func[channel][i] = temp_y;

        // �DM[0]-0�����
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i <= mapping_Func[channel][0]; i++)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * i;
        }
        mapping_Func[channel][0] = sum2 / sum1;

        // �DM[255]-255�����
        sum1 = 0, sum2 = 0;
        for (int i = 255; i >= mapping_Func[channel][255]; i--)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * i;
        }
        mapping_Func[channel][255] = sum2 / sum1;

        //  mapping function.
        /*int n = 0;
        cout << "channel: " << channel << endl;
        for (auto& item : mapping_Func[channel])
        {
            cout << n << " -> " << item << endl;
            n++;
        }*/
    }

    Mat output_comp = Mat(ref.size(), ref.type());

    for (int channel = 0; channel < 3; channel++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                if (tar_mask.at<bool>(i, j) == true)
                    output_comp.at<Vec3b>(i, j)[channel] = mapping_Func[channel][tar.at<Vec3b>(i, j)[channel]];
                else
                    output_comp.at<Vec3b>(i, j)[channel] = tar.at<Vec3b>(i, j)[channel];

    tar.convertTo(tar, CV_64FC3);
    output_comp.convertTo(output_comp, CV_64FC3);
    output_comp -= tar;
    tar.convertTo(tar, CV_8UC3);

    return output_comp;
}

Mat Utils::HM_Fecker_corr(Mat &ref, Mat &tar, Mat &overlap, Mat &tar_mask)
{
    vector<Mat> overlap_correlation;
    switch (corr_status)
    {
    case 0:
        break;
    case 1:
        overlap_correlation = Utils::correlation_coef_estimate(ref, tar, overlap);
        break;
    case 2:
        overlap_correlation = Utils::correlation_cos_estimate(ref, tar, overlap);
        break;
    case 3:
        overlap_correlation = Utils::correlation_cos2_estimate(ref, tar, overlap);
        break;
    default:
        break;
    }

    // for (int i = 0; i < 3; i++)
    //  {
    //      if (corr_status == 0) break;
    //      imwrite("data/"+to_string(i)+"_corre.png", overlap_correlation[i]);
    //  }

    vector<int> mapping_Func[3];

    for (int channel = 0; channel < 3; channel++)
    {
        mapping_Func[channel].assign(256, 0);

        ImgPack ref_DF, tar_DF;
        ref_DF = CalDF(ref, channel, overlap_correlation[channel]);
        tar_DF = CalDF(tar, channel, overlap_correlation[channel]);

        // �u�n�O���Ĥ@�ӡA�ҥH�ݭn�@�Ӷ}���A�����Ĥ@�ӥ��`�M�g�Υ��`�Q�M�g����
        bool flag = false;
        int temp_x = -100, temp_y = -100;

        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                if (ref_DF.CDF[j] > tar_DF.CDF[i])
                {
                    if (flag == false && (j - 1) >= 0)
                    {
                        flag = true; // �}�����}!
                        temp_x = i;
                        temp_y = j;
                    }

                    // saturate_cast<uchar>�i�H�T�O�ƭȦbuchar 8bits��(0-255).
                    mapping_Func[channel][i] = (int)saturate_cast<uchar>(j);
                    break;
                }

                // ���L�ȥi�M�g�ɨ��e�@�ӬM�g��
                mapping_Func[channel][i] = mapping_Func[channel][i - 1];
            }
        }

        // cout << "tempx,y: " << temp_x << ' ' << temp_y << endl;

        // ���L�ȥi�M�g�ɨ��Ĥ@�ӥ��`�M�g����
        for (int i = temp_x; i >= 0; i--)
            mapping_Func[channel][i] = temp_y;

        // �DM[0]-0�����
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i <= mapping_Func[channel][0]; i++)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * i;
        }
        mapping_Func[channel][0] = sum2 / sum1;

        // �DM[255]-255�����
        sum1 = 0, sum2 = 0;
        for (int i = 255; i >= mapping_Func[channel][255]; i--)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * i;
        }
        mapping_Func[channel][255] = sum2 / sum1;
    }

    Mat output_comp = Mat(ref.size(), ref.type());

    for (int channel = 0; channel < 3; channel++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
                // if (tar_mask.at<bool>(i, j) == true)
                bool sw = tar.at<Vec3b>(i, j)[0] != 0 || tar.at<Vec3b>(i, j)[1] != 0 || tar.at<Vec3b>(i, j)[2] != 0;
                if (sw)
                    output_comp.at<Vec3b>(i, j)[channel] = mapping_Func[channel][tar.at<Vec3b>(i, j)[channel]];
                else
                    output_comp.at<Vec3b>(i, j)[channel] = tar.at<Vec3b>(i, j)[channel];
            }

    tar.convertTo(tar, CV_64FC3);
    output_comp.convertTo(output_comp, CV_64FC3);
    output_comp -= tar;
    tar.convertTo(tar, CV_8UC3);

    return output_comp;
}

vector<Mat> Utils::correlation_coef_estimate(Mat &ref, Mat &tar, Mat &overlap)
{
    vector<Mat> overlap_correlation;

    for (int i = 0; i < 3; i++)
        overlap_correlation.push_back(Mat::zeros(overlap.size(), overlap.type()));

    vector<Point> location;

    int min_x = height, min_y = width, max_x = -1, max_y = -1;
    findNonZero(overlap, location);
    for (auto &item : location)
    {
        min_x = item.x < min_x ? item.x : min_x;
        min_y = item.y < min_y ? item.y : min_y;
        max_x = item.x > max_x ? item.x : max_x;
        max_y = item.y > max_y ? item.y : max_y;
    }
    cout << "corre: " << min_x << ", " << min_y << ", " << max_x << ", " << max_y << endl;

    for (int x = min_y; x <= max_y - correlation_block + 1; x += correlation_block)
    {
        for (int y = min_x; y <= max_x - correlation_block + 1; y += correlation_block)
        {
            vector<double> mean_x(3, 0), mean_y(3, 0);
            for (int i = 0; i < correlation_block; i++)
                for (int j = 0; j < correlation_block; j++)
                    for (int c = 0; c < 3; c++)
                    {
                        mean_x[c] += ref.at<Vec3b>(x + i, y + j)[c];
                        mean_y[c] += tar.at<Vec3b>(x + i, y + j)[c];
                    }

            for (auto &item : mean_x)
                item /= pow(correlation_block, 2);
            for (auto &item : mean_y)
                item /= pow(correlation_block, 2);

            vector<double> p(3, 0);
            vector<double> numerator(3, 0), denominator_x(3, 0), denominator_y(3, 0);
            for (int i = 0; i < correlation_block; i++)
                for (int j = 0; j < correlation_block; j++)
                    for (int c = 0; c < 3; c++)
                    {
                        numerator[c] += (1.0 * ref.at<Vec3b>(x + i, y + j)[c] - mean_x[c]) * (1.0 * tar.at<Vec3b>(x + i, y + j)[c] - mean_y[c]);
                        denominator_x[c] += pow(1.0 * ref.at<Vec3b>(x + i, y + j)[c] - mean_x[c], 2);
                        denominator_y[c] += pow(1.0 * tar.at<Vec3b>(x + i, y + j)[c] - mean_y[c], 2);
                    }

            for (int c = 0; c < 3; c++)
            {
                p[c] = numerator[c] / pow(denominator_x[c] * denominator_y[c], 0.5);
                if (p[c] >= correlation_TH)
                {
                    for (int i = 0; i < correlation_block; i++)
                        for (int j = 0; j < correlation_block; j++)
                        {
                            overlap_correlation[c].at<uchar>(x + i, y + j) = 255;
                        }
                }
            }
        }
    }

    for (int c = 0; c < 3; c++)
    {
        bitwise_and(overlap_correlation[c], overlap, overlap_correlation[c]);
        // imwrite("data/" + to_string(c)+".png", overlap_correlation[c]);
    }

    return overlap_correlation;
}

vector<Mat> Utils::correlation_cos_estimate(Mat &ref, Mat &tar, Mat &overlap)
{
    vector<Mat> overlap_correlation;

    for (int i = 0; i < 3; i++)
        overlap_correlation.push_back(Mat::zeros(overlap.size(), overlap.type()));

    vector<Point> location;

    int min_x = height, min_y = width, max_x = -1, max_y = -1;
    findNonZero(overlap, location);
    for (auto &item : location)
    {
        min_x = item.x < min_x ? item.x : min_x;
        min_y = item.y < min_y ? item.y : min_y;
        max_x = item.x > max_x ? item.x : max_x;
        max_y = item.y > max_y ? item.y : max_y;
    }
    cout << "corre: " << min_x << ", " << min_y << ", " << max_x << ", " << max_y << endl;

    for (int x = min_y; x <= max_y - correlation_block + 1; x += correlation_block)
    {
        for (int y = min_x; y <= max_x - correlation_block + 1; y += correlation_block)
        {
            vector<double> mean_x(3, 0), mean_y(3, 0);
            for (int i = 0; i < correlation_block; i++)
                for (int j = 0; j < correlation_block; j++)
                    for (int c = 0; c < 3; c++)
                    {
                        mean_x[c] += ref.at<Vec3b>(x + i, y + j)[c];
                        mean_y[c] += tar.at<Vec3b>(x + i, y + j)[c];
                    }

            for (auto &item : mean_x)
                item /= pow(correlation_block, 2);
            for (auto &item : mean_y)
                item /= pow(correlation_block, 2);

            vector<double> p(3, 0);
            vector<double> std_x(3, 0), std_y(3, 0);
            for (int i = 0; i < correlation_block; i++)
                for (int j = 0; j < correlation_block; j++)
                    for (int c = 0; c < 3; c++)
                    {
                        std_x[c] += pow(1.0 * ref.at<Vec3b>(x + i, y + j)[c] - mean_x[c], 2);
                        std_y[c] += pow(1.0 * tar.at<Vec3b>(x + i, y + j)[c] - mean_y[c], 2);
                    }

            for (int c = 0; c < 3; c++)
            {
                // cos theta
                // inner product
                p[c] = mean_x[c] * mean_y[c] + std_x[c] * std_y[c];
                // cross product
                p[c] /= pow(pow(mean_x[c], 2) + pow(std_x[c], 2), 0.5) * pow(pow(mean_y[c], 2) + pow(std_y[c], 2), 0.5);

                if (p[c] >= correlation_TH)
                {
                    for (int i = 0; i < correlation_block; i++)
                        for (int j = 0; j < correlation_block; j++)
                        {
                            overlap_correlation[c].at<uchar>(x + i, y + j) = 255;
                        }
                }
            }
        }
    }

    for (int c = 0; c < 3; c++)
    {
        bitwise_and(overlap_correlation[c], overlap, overlap_correlation[c]);
        // imwrite("data/" + to_string(c)+".png", overlap_correlation[c]);
    }

    return overlap_correlation;
}

vector<Mat> Utils::correlation_cos2_estimate(Mat &ref, Mat &tar, Mat &overlap)
{
    vector<Mat> overlap_correlation;

    for (int i = 0; i < 3; i++)
        overlap_correlation.push_back(Mat::zeros(overlap.size(), overlap.type()));

    vector<Point> location;

    int min_x = height, min_y = width, max_x = -1, max_y = -1;
    findNonZero(overlap, location);
    for (auto &item : location)
    {
        min_x = item.x < min_x ? item.x : min_x;
        min_y = item.y < min_y ? item.y : min_y;
        max_x = item.x > max_x ? item.x : max_x;
        max_y = item.y > max_y ? item.y : max_y;
    }
    cout << "corre: " << min_x << ", " << min_y << ", " << max_x << ", " << max_y << endl;

    for (int x = min_y; x <= max_y - correlation_block + 1; x += correlation_block)
    {
        for (int y = min_x; y <= max_x - correlation_block + 1; y += correlation_block)
        {
            vector<double> p(3, 0);
            vector<vector<double>> vec_x(3), vec_y(3);
            for (int i = 0; i < correlation_block; i++)
                for (int j = 0; j < correlation_block; j++)
                    for (int c = 0; c < 3; c++)
                    {
                        vec_x[c].push_back(1.0 * ref.at<Vec3b>(x + i, y + j)[c]);
                        vec_y[c].push_back(1.0 * tar.at<Vec3b>(x + i, y + j)[c]);
                    }

            for (int c = 0; c < 3; c++)
            {
                // cos theta
                // inner product
                p[c] = inner_product(vec_x[c].begin(), vec_x[c].end(), vec_y[c].begin(), 0);
                // cross product
                p[c] /= norm(vec_x[c]) * norm(vec_y[c]);

                if (p[c] >= correlation_TH)
                {
                    for (int i = 0; i < correlation_block; i++)
                        for (int j = 0; j < correlation_block; j++)
                        {
                            overlap_correlation[c].at<uchar>(x + i, y + j) = 255;
                        }
                }
            }
        }
    }

    for (int c = 0; c < 3; c++)
    {
        bitwise_and(overlap_correlation[c], overlap, overlap_correlation[c]);
        // imwrite("data/" + to_string(c)+".png", overlap_correlation[c]);
    }

    return overlap_correlation;
}

void Inits::build_exe_sequence()
{
    vector<double> seq;
    Mat mask_tmp;
    int N = warped_imgs.size();

    for (int i = 0; i < N; i++)
    {
        // seq.push_back(Utils::Thenengrad(warped_imgs[i]));
        seq.push_back(Utils::laplacian(warped_imgs[i]));
    }

    vector<int> idx;
    for (int i = 0; i < N; i++)
        idx.push_back(i);
    switch (exe_seq_sw)
    {
    case 1:
        sort(idx.begin(), idx.end(), [seq](int x, int y)
             { return seq[x] > seq[y]; });
        break;
    case 2:
        sort(idx.begin(), idx.end(), [seq](int x, int y)
             { return seq[x] < seq[y]; });
        break;
    default:
        break;
    }
    for (auto &item : seq)
        cout << item << " ";

    exe_seq = idx;
}

double Utils::Thenengrad(Mat &img)
{
    assert(img.empty());

    Mat gray_img, sobel_x, sobel_y, G;
    if (img.channels() == 3)
    {
        cvtColor(img, gray_img, COLOR_BGR2GRAY);
        Sobel(gray_img, sobel_x, CV_32FC1, 1, 0);
        Sobel(gray_img, sobel_y, CV_32FC1, 0, 1);
        multiply(sobel_x, sobel_x, sobel_x);
        multiply(sobel_y, sobel_y, sobel_y);
        Mat sqrt_mat = sobel_x + sobel_y;
        sqrt(sqrt_mat, G);
    }

    return mean(G)[0];
}

double Utils::laplacian(Mat &img)
{
    assert(img.empty());

    Mat gray_img, lap_img;

    if (img.channels() == 3)
    {
        cvtColor(img, gray_img, COLOR_BGR2GRAY);
        Laplacian(gray_img, lap_img, CV_32FC1);
        lap_img = abs(lap_img);
    }

    // return mean(lap_img)[0];
    vector<Point> locations;
    findNonZero(gray_img, locations);
    return double(sum(lap_img)[0]) / locations.size();
}

void Utils::EF_based()
{
    check_corrected.resize(warped_imgs.size(), false);
    for (auto &item : exe_seq)
        cout << "\nexe_seq: " << item << " ";
    cout << endl;
    
    for (int &i : exe_seq)
    {
        bool check = false;
        for (int j = 0; j < ref_seq[i].size(); j++)
        {
            if (check_corrected[ref_seq[i][j]] == true)
            {
                check = true;
                break;
            }
        }

        for (int j = 0; j < ref_seq[i].size(); j++)
        {
            if (refercon)
                if (check == true && check_corrected[ref_seq[i][j]] == false)
                    continue;
            Inits::Initialize_var();
            int me = 0;
            for (int find_me = 0; find_me < ref_seq[ref_seq[i][j]].size(); find_me++)
                if (ref_seq[ref_seq[i][j]][find_me] == i)
                {
                    me = find_me;
                    break;
                }

            cout << "what wrong? " << i << " " << ref_seq[i][j] << ", check: " << check << endl;
            // fusion color transfer
            // AJBI-based
            seam_pixel_lst = Utils::build_seam_pixel_lst(seams[i][j]);
            bool ismerge = Utils::refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(seam_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst, warped_imgs[ref_seq[i][j]], warped_imgs[i]);
            for (Point p : discarded_seam_pixel_lst)
            {
                anomaly_mask.at<uchar>(p) = 255;
            }
            Utils::sort_seam_pixel_lst(seam_pixel_lst, sorted_seam_pixel_lst, seams[i][j]);
            Utils::build_range_map_with_side_addition(sorted_seam_pixel_lst, masks[i][j], warped_imgs[i]);
            // fusion2 weight(wavefront)
            wave_num.push_back(discovered_time_stamp_map);
            discovered_time_stamp_map.copyTo(wave_num[wave_num.size() - 1]);

            // fusion1 weight(cite_range)
            cites_range.push_back(range_count_map);
            range_count_map.copyTo(cites_range[cites_range.size() - 1]);

            target_pixel_lst = Utils::build_target_pixel_lst(masks[i][j], seams[i][j], warped_imgs[i]);
            Utils::init_color_comp_map(seam_pixel_lst, warped_imgs[ref_seq[i][j]], warped_imgs[i]);

            vector<vector<Point>> pixel_wise_lst;
            for (Point p : target_pixel_lst)
            {
                vector<Point> a;
                a.push_back(p);
                pixel_wise_lst.push_back(a);
            }
            Utils::update_color_comp_map_range_anomaly__parallel_ver(pixel_wise_lst, warped_imgs[i]);
            comps.push_back(Mat::zeros(height, width, CV_64FC3));
            color_comp_map.copyTo(comps[comps.size() - 1]);

            // HM-based
            Fecker_comps.push_back(Mat::zeros(height, width, CV_64FC3));
            Fecker_color_comp_map = Utils::HM_Fecker_corr(warped_imgs[ref_seq[i][j]], warped_imgs[i], overlaps[i][j], masks[i][j]);
            Fecker_color_comp_map.copyTo(Fecker_comps[Fecker_comps.size() - 1]);
        }

        check_corrected[i] = true;

        // Utils::build_final_result_Before(warped_imgs[i], i);
        Utils::build_final_result_Before_with_Fecker_weighting(warped_imgs[i], i);
        comps.erase(comps.begin(), comps.end());
        Fecker_comps.erase(Fecker_comps.begin(), Fecker_comps.end());
        cites_range.erase(cites_range.begin(), cites_range.end());
        wave_num.erase(wave_num.begin(), wave_num.end());
    }
}

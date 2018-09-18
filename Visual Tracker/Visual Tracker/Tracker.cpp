
#include"stdafx.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<core.hpp>
#include<imgproc.hpp>
#include<string>


#include"Tracker.h"
#include"fhog.h"

void KCF::run(Video *seq)const
{
	//std::cout << "test complete!" << std::endl;
	cv::namedWindow("Tracking Video", cv::WINDOW_AUTOSIZE);

	cv::Point target_pos;
	target_pos.x = seq->ground_truth[0].x;
	target_pos.y = seq->ground_truth[0].y;
	cv::Size target_sz;
	target_sz.width = seq->ground_truth[0].w;
	target_sz.height = seq->ground_truth[0].h;

	cv::Size window_size;
	window_size.width = target_sz.width*padding;
	window_size.height = target_sz.height*padding;

	float output_sigma = sqrt(target_sz.width*target_sz.height)*output_sigma_factor / cell_size;
	cv::Size use_sz;
	use_sz.width = window_size.width / cell_size;
	use_sz.height = window_size.height / cell_size;
	cv::Mat y = CreateGaussian2D(use_sz, output_sigma);
	cv::Mat yf;
	cv::dft(y, yf, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat cos_window(use_sz, CV_32F);
	cos_window = hann(use_sz.width)*hann(use_sz.height).t();

	int video_len = seq->length;

	//test
	cv::String path = seq->img_path[0];
	cv::Mat img = cv::imread(path);
	cv::imshow("1",img);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	cv::imshow("2", img);

	cv::Mat patch(window_size, CV_32F);
	patch= get_subwindow(img, target_pos, window_size);

	int w = patch.cols, h = patch.rows;
	float* I = new float[h*w];
	const float* ptr_img = patch.ptr<float>();
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			//注意点，灰度图像应为uchar
			I[i*h + j] = (float)patch.at<uchar>(j, i);

		}
	}

	cv::imshow("Tracking Video", patch);
	cvvWaitKey(0);
	//test end


	/*
	for (int frame = 0; frame < video_len; frame++) {
		cv::String path = seq->img_path[frame];
		cv::Mat img = cv::imread(path);

		cv::imshow("Tracking Video", img);

		if (img.dims > 1) {
			cv::Mat img_temp = img;
			cv::cvtColor(img_temp, img, CV_RGB2GRAY);
		}

		cv::Mat model_alphaf;
		std::vector<cv::Mat> model_xf;
		if (frame > 0) {
			cv::Mat patch = get_subwindow(img, target_pos, window_size);
			std::vector<cv::Mat> x = get_features(patch, cos_window);
			std::vector<cv::Mat> xf(x.size());
			for (int i = 0; i < x.size(); i++)
				cv::dft(x[i], xf[i], cv::DFT_COMPLEX_OUTPUT);
			cv::Mat kzf = Gaussian_kernel(xf, model_xf);
			cv::Mat responsef = model_alphaf.mul(kzf);
			cv::Mat response;
			cv::idft(responsef, response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

			cv::Point delta = find_max(response);

			target_pos = target_pos + cell_size * (delta);

		}

		cv::Mat patch = get_subwindow(img, target_pos, window_size);

		std::vector<cv::Mat> x = get_features(patch, cos_window);
		std::vector<cv::Mat> xf(x.size());
		for (int i = 0; i < x.size(); i++) {
			cv::dft(x[i], xf[i], cv::DFT_COMPLEX_OUTPUT);
		}

		cv::Mat kf = Gaussian_kernel(xf, xf);

		cv::Mat alpha_f = div_pointwise(yf, (kf + lambda));


		if (frame == 0) {
			model_alphaf = alpha_f;
			model_xf = xf;
		}
		else {
			model_alphaf = (1 - interp_factor)*model_alphaf + interp_factor * alpha_f;
			int len = model_xf.size();
			for (int i = 0; i < len; i++)
				model_xf[i] = (1 - interp_factor)*model_xf[i] + interp_factor * xf[i];
		}

		cv::imshow("Tracking Video", img);

	}*/
}

cv::Mat KCF::CreateGaussian1D(int len, float sigma)const
{
	cv::Mat label(len, 1, CV_32F);
	float* label_ptr = label.ptr<float>();
	double scale = -0.5 / (sigma*sigma);

	for (int i = 0; i < len; i++) {
		double x = i + 1 - floor(len / 2);
		double value = std::exp(scale*x*x);
		label_ptr[i] = value;
	}
	return label;
}

cv::Mat KCF::CreateGaussian2D(const cv::Size &sz,const  float &sigma)const
{
	cv::Mat a = CreateGaussian1D(sz.width, sigma);
	cv::Mat b = CreateGaussian1D(sz.height, sigma);

	cv::Mat lable = a * b.t();
	return lable;
}

cv::Mat KCF::GetGaussianSharpLabels(const cv::Size &sz, const float &sigma) const
{
	cv::Mat label = CreateGaussian2D(sz, sigma);

	cv::Size circ_V = cv::Size(-(floor(sz.width / 2) + 1), -(floor(sz.height / 2) + 1));

	label = CircShift(label, circ_V);

	return label;
}

cv::Mat KCF::CircShift(const cv::Mat &src, const cv::Size &V) const
{
	cv::Mat res(src.size(), CV_32F);

	int h, w;
	if (V.width < 0) {
		w = -V.width;
	}
	else {
		w = src.cols - V.width;
	}
	if (V.height < 0) {
		h = -V.height;
	}
	else {
		h = src.rows - V.height;
	}

	cv::Mat a1(src, cv::Rect(0, 0, w, h));
	cv::Mat a2(src, cv::Rect(w, 0, src.cols - w, h));
	cv::Mat a3(src, cv::Rect(0, h, w, src.rows - h));
	cv::Mat a4(src, cv::Rect(w, h, src.cols - w, src.cols - h));

	cv::Mat temp1, temp2;
	cv::hconcat(a4, a3, temp1);
	cv::hconcat(a2, a1, temp2);
	cv::vconcat(temp1, temp2, res);


	return res;
}

cv::Mat KCF::hann(int len) const
{
	cv::Mat res(len, 1, CV_32F);

	float* ptr = res.ptr<float>();

	float PI = asin(1) * 2;

	for (int i = 0; i < len; i++) {
		ptr[i] = (1 - cos(2 * PI*i) / (len - 1)) / 2.;
	}

	return res;
}

cv::Mat KCF::get_subwindow(const cv::Mat &img, cv::Point pos, cv::Size sz) const
{
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))
	cv::Point lefttop(pos.x - floor(sz.width / 2), pos.y - floor(sz.height / 2));
	cv::Point rightbottom(pos.x + sz.width - floor(sz.width / 2), pos.y + sz.height - floor(sz.height / 2));
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0), max(rightbottom.x - img.cols - 1, 0), max(rightbottom.y - img.rows - 1,0));
	cv::Point r_lefttop(max(lefttop.x, 0), max(lefttop.y, 0));
	cv::Point r_rightbottom(min(rightbottom.x, img.cols), min(rightbottom.y, img.rows));
	cv::Rect r_rect(r_lefttop, r_rightbottom);

	cv::Mat res(r_rect.size(), CV_32F);
	img(r_rect).copyTo(res);
	cv::copyMakeBorder(res, res, border.x, border.y, border.width, border.height, cv::BORDER_REPLICATE);

#undef min(a,b)
#undef max(a,b)

	return res;
}

std::vector<cv::Mat> KCF::get_features(const cv::Mat &img, cv::Mat cos_window) const
{
	std::vector<cv::Mat> feature;

	if (feature_type == "Hog") {
		feature = get_hog(img);
	}
	else if (feature_type == "Gray") {
		feature.push_back(img.clone());
	}

	int dim = feature.size();
	for (int i = 0; i < dim; i++)
		feature[i] = feature[i].mul(cos_window);

	return feature;
}

std::vector<cv::Mat> KCF::get_hog(const cv::Mat &img) const
{
	std::vector<cv::Mat> res;

	res = fhog(img);

	return cv::Mat();
}

cv::Mat KCF::Gaussian_kernel(const std::vector<cv::Mat> &xf, const std::vector<cv::Mat> &yf) const
{
	cv::Size sz = xf[0].size();
	int N = sz.width*sz.height;
	int dim = xf.size();
	double xx = 0, yy = 0;
	cv::Mat xyf, _xy, xy;
	for (int i = 0; i < dim; i++) {
		xx += cv::norm(xf[i])*cv::norm(xf[i]) / N;
		yy += cv::norm(yf[i])*cv::norm(yf[i]) / N;
		cv::mulSpectrums(xf[i], xf[i], xyf, 0, true);
		cv::idft(xyf, _xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		xy += _xy;
	}
	cv::Mat k, kf;
	exp((-1 / kernel_sigma * kernel_sigma*cv::max(0, (xx + yy - 2 * xy) / (N*dim))), k);
	cv::dft(k, kf, cv::DFT_COMPLEX_OUTPUT);
	return kf;
}

cv::Mat KCF::div_pointwise(const cv::Mat &x,const cv::Mat &y) const
{
	cv::Mat res;
	std::vector<cv::Mat> plane1;
	cv::split(x, plane1);
	std::vector<cv::Mat> plane2;
	cv::split(y, plane2);
	cv::Mat c2_d2 = plane2[0].mul(plane2[0]) + plane2[1].mul(plane2[1]);
	std::vector<cv::Mat> res_t;
	res_t[0] = plane1[0].mul(plane2[0]) + plane1[1].mul(plane2[1]);
	res_t[0] = res_t[0] / c2_d2;
	res_t[1] = plane1[1].mul(plane2[0]) - plane1[0].mul(plane2[1]);
	res_t[1] = res_t[1] / c2_d2;
	cv::merge(res_t, res);

	return res;
}

cv::Point KCF::find_max(const cv::Mat &x) const
{
	cv::Point res;

	const float* x_ptr = x.ptr<float>();
	float max_x, max_y, max_v = 0x8000000;
	int m = x.cols, n = x.rows;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (x_ptr[i*n+j] > max_v) {
				max_v = x_ptr[i*n + j];
				max_x = i;
				max_y = j;
			}
		}
	}

	return cv::Point(max_x, max_y);
}
